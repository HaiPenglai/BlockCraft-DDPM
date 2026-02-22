import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import UNet2DConditionModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm.auto import tqdm

# --- 配置 ---
class Config:
    # 数据集路径
    single_dir = "dataset"
    pair_dir = "dataset_pairs"
    output_dir = "mc_block_model_conditional"
    image_size = 128
    
    # 5090 优化参数
    train_batch_size = 128
    learning_rate = 2e-4
    num_epochs = 300  # 增加轮数以增强泛化性推理能力
    save_every_epochs = 20
    mixed_precision = "bf16" 
    num_workers = 8

    # 方块名称到 ID 的映射
    block_map = {
        "none": 0,
        "brick": 1,
        "cobblestone": 2,
        "diamond_ore": 3,
        "emerald_block": 4,
        "planks_oak": 5
    }

# --- 数据集处理 ---
class MCBlockDataset(Dataset):
    def __init__(self, single_root, pair_root, transform, block_map):
        self.samples = []
        self.transform = transform
        self.block_map = block_map

        # 1. 扫描单方块 (Label: [ID, 0])
        for folder in os.listdir(single_root):
            if folder in block_map:
                folder_path = os.path.join(single_root, folder)
                for img_name in os.listdir(folder_path):
                    if img_name.lower().endswith(('.png', '.jpg')):
                        self.samples.append({
                            "path": os.path.join(folder_path, img_name),
                            "labels": [block_map[folder], 0] 
                        })

        # 2. 扫描双方块 (Label: [ID_Left, ID_Right])
        for folder in os.listdir(pair_root):
            if "_and_" in folder:
                parts = folder.split("_and_")
                if parts[0] in block_map and parts[1] in block_map:
                    folder_path = os.path.join(pair_root, folder)
                    for img_name in os.listdir(folder_path):
                        if img_name.lower().endswith(('.png', '.jpg')):
                            self.samples.append({
                                "path": os.path.join(folder_path, img_name),
                                "labels": [block_map[parts[0]], block_map[parts[1]]]
                            })

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        image = Image.open(sample["path"]).convert("RGB")
        # 将 labels 转为 tensor
        return self.transform(image), torch.tensor(sample["labels"], dtype=torch.long)

# --- 训练逻辑 ---
def train():
    config = Config()
    accelerator = Accelerator(mixed_precision=config.mixed_precision)

    # 1. 定义有条件 UNet
    # 我们使用 Cross-Attention 维度 128 来接收语义 Embedding
    model = UNet2DConditionModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        cross_attention_dim=128, # 语义向量维度
    )

    # 2. 语义 Embedding 层：将 ID 映射为向量
    # 6 个方块(含none) x 128维度
    label_emb = nn.Embedding(6, 128) 

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(label_emb.parameters()), lr=config.learning_rate)

    preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    dataset = MCBlockDataset(config.single_dir, config.pair_dir, preprocess, config.block_map)
    train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    model, label_emb, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, label_emb, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        os.makedirs("samples", exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)

    # 训练循环
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        
        for step, (images, labels) in enumerate(train_dataloader):
            # labels shape: [BS, 2]
            
            # 将 [左, 右] 标签转为 Embedding
            # [BS, 2] -> [BS, 2, 128]
            encoder_hidden_states = label_emb(labels)

            noise = torch.randn(images.shape).to(images.device)
            timesteps = torch.randint(0, 1000, (images.shape[0],), device=images.device).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            # 预测噪声时传入语义 Embedding
            noise_pred = model(noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred, noise)
            
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

        # 定期保存与【测试泛化性】采样
        if (epoch + 1) % config.save_every_epochs == 0 and accelerator.is_main_process:
            model.eval()
            print(f"\nEpoch {epoch+1}: 正在测试泛化性采样...")
            
            # 测试用例：1.钻石单方块  2.训练过组合  3.从未见过的组合 (砖块+橡木)
            test_prompts = [
                [3, 0], # diamond single
                [3, 4], # diamond + emerald (Seen)
                [1, 5], # brick + oak (假设这个没见过)
                [3, 2]  # diamond + cobblestone (泛化性关键测试)
            ]
            
            with torch.no_grad():
                for idx, p in enumerate(test_prompts):
                    p_tensor = torch.tensor([p], device=accelerator.device)
                    emb = label_emb(p_tensor)
                    
                    # 采样循环
                    sample = torch.randn(1, 3, 128, 128).to(accelerator.device)
                    for t in tqdm(noise_scheduler.timesteps, leave=False):
                        residual = model(sample, t, encoder_hidden_states=emb).sample
                        sample = noise_scheduler.step(residual, t, sample).prev_sample
                    
                    # 保存
                    sample = ((sample / 2 + 0.5).clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy().astype("uint8")
                    Image.fromarray(sample[0]).save(f"samples/epoch_{epoch+1}_test_{idx}.png")
            
            # 保存模型
            torch.save({
                'unet': accelerator.unwrap_model(model).state_dict(),
                'emb': accelerator.unwrap_model(label_emb).state_dict(),
                'config': config.block_map
            }, f"{config.output_dir}/latest_model.pt")

if __name__ == "__main__":
    train()