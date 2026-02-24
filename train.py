# --- START OF FILE train.py ---
import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm.auto import tqdm

# --- 配置 ---
class Config:
    dataset_dir = "dataset"
    output_dir = "mc_blocks_ddpm_cond" # 类似四叶草的命名风格
    image_size = 128
    
    # 训练参数
    train_batch_size = 64  # RTX 5090 显存非常大，64或128都可以轻松应对
    learning_rate = 1e-4
    num_epochs = 200       # 遵循你之前成功的200轮经验
    save_every_epochs = 10 # 每10轮保存并测试
    mixed_precision = "bf16" # RTX 5090 支持 bf16，速度更快
    num_workers = 8

    # CFG (Classifier-Free Guidance) 训练配置
    cfg_drop_rate = 0.1    # 10% 的概率丢弃条件，用于训练无条件生成能力
    cross_attention_dim = 128

# --- 数据集处理 (动态读取目录) ---
class MCBlockDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.samples = []
        self.transform = transform
        
        # 动态获取所有类别
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_id = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            folder_path = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg')):
                    self.samples.append({
                        "path": os.path.join(folder_path, img_name),
                        "label_id": self.class_to_id[cls_name]
                    })
        print(f"✅ 成功加载数据集，共 {len(self.classes)} 个类别，{len(self.samples)} 张图片。")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        image = Image.open(sample["path"]).convert("RGB")
        return self.transform(image), torch.tensor(sample["label_id"], dtype=torch.long)

# --- 训练逻辑 ---
def train():
    config = Config()
    accelerator = Accelerator(mixed_precision=config.mixed_precision)

    # 1. 数据集准备
    preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = MCBlockDataset(config.dataset_dir, preprocess)
    train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers)
    num_classes = len(dataset.classes)

    # 保存类别映射，供 generate.py 使用
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs("samples_mc", exist_ok=True)
        with open(os.path.join(config.output_dir, "class_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(dataset.class_to_id, f, ensure_ascii=False, indent=4)

    # 2. 定义有条件 UNet
    model = UNet2DConditionModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        cross_attention_dim=config.cross_attention_dim, 
    )
    
    # 🌟 核心要求：启用梯度检查点，拯救/优化显存
    model.enable_gradient_checkpointing() 

    # 3. 语义 Embedding 层 (重点：多加1个ID给“空条件”作为CFG的无条件引导)
    # index 0 ~ num_classes-1 是真实类别，index = num_classes 是无条件(uncond)
    label_emb = nn.Embedding(num_classes + 1, config.cross_attention_dim) 

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(label_emb.parameters()), lr=config.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    model, label_emb, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, label_emb, optimizer, train_dataloader, lr_scheduler
    )

    # 训练循环
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}/{config.num_epochs}")
        
        for step, (images, labels) in enumerate(train_dataloader):
            # --- CFG 训练核心逻辑 ---
            # 有 config.cfg_drop_rate 的概率，把原本的标签替换为 num_classes (即无条件类别)
            drop_mask = torch.rand(labels.shape, device=labels.device) < config.cfg_drop_rate
            cfg_labels = torch.where(drop_mask, torch.tensor(num_classes, device=labels.device), labels)
            
            # UNet2DConditionModel 需要的维度是 (BS, Seq_Len, Dim)，因此增加一个 Seq_Len=1 的维度
            encoder_hidden_states = label_emb(cfg_labels).unsqueeze(1) # Shape: [BS, 1, 128]

            noise = torch.randn_like(images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=images.device).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            # 预测噪声
            noise_pred = model(noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred, noise)
            
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

        # 🌟 每 10 轮保存与【测试采样】
        if (epoch + 1) % config.save_every_epochs == 0 and accelerator.is_main_process:
            model.eval()
            print(f"\n✨ Epoch {epoch+1}: 正在进行 CFG 条件生成测试...")
            
            # 随机抽2个类别测试
            test_classes = random.sample(dataset.classes, 2)
            
            with torch.no_grad():
                for test_cls in test_classes:
                    target_id = dataset.class_to_id[test_cls]
                    
                    # 准备 CFG 的条件和无条件 Embedding
                    cond_id = torch.tensor([target_id], device=accelerator.device)
                    uncond_id = torch.tensor([num_classes], device=accelerator.device) # 空条件
                    
                    cond_emb = label_emb(cond_id).unsqueeze(1)    # [1, 1, 128]
                    uncond_emb = label_emb(uncond_id).unsqueeze(1) # [1, 1, 128]
                    
                    # 采样起点
                    sample = torch.randn(1, 3, config.image_size, config.image_size).to(accelerator.device)
                    
                    # 为了测试速度快，设置 50 步采样
                    noise_scheduler.set_timesteps(50)
                    guidance_scale = 3.0 # CFG 引导系数
                    
                    for t in tqdm(noise_scheduler.timesteps, leave=False, desc=f"Generating {test_cls}"):
                        # CFG 推理：同时计算条件和无条件的噪声预测 (合并为一个 Batch 提高效率)
                        latent_model_input = torch.cat([sample] * 2)
                        emb_input = torch.cat([uncond_emb, cond_emb])
                        
                        noise_pred = model(latent_model_input, t, encoder_hidden_states=emb_input).sample
                        
                        # 拆分预测结果并运用 CFG 公式
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        
                        sample = noise_scheduler.step(noise_pred_cfg, t, sample).prev_sample
                    
                    # 保存图片
                    sample_img = ((sample / 2 + 0.5).clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy().astype("uint8")[0]
                    Image.fromarray(sample_img).save(f"samples_mc/epoch_{epoch+1:03d}_{test_cls}.png")
            
            # 保存该轮的模型
            checkpoint_dir = os.path.join(config.output_dir, f"epoch_{epoch+1:03d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'unet': accelerator.unwrap_model(model).state_dict(),
                'emb': accelerator.unwrap_model(label_emb).state_dict(),
                'num_classes': num_classes
            }, os.path.join(checkpoint_dir, "model.pt"))
            print(f"✅ 模型已保存至 {checkpoint_dir}")

if __name__ == "__main__":
    train()
# --- END OF FILE train.py ---