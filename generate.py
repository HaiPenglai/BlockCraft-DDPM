import os
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler
from PIL import Image
from tqdm.auto import tqdm
import numpy as np

# --- 1. 配置与参数 (需与训练时保持一致) ---
MODEL_PATH = "mc_block_model_conditional/latest_model.pt" # 你的模型路径
OUTPUT_DIR = "final_results"
IMAGE_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 必须与训练时的 block_map 完全一致
BLOCK_MAP = {
    "none": 0,
    "brick": 1,
    "cobblestone": 2,
    "diamond_ore": 3,
    "emerald_block": 4,
    "planks_oak": 5
}
# 反向映射，方便打印
ID_TO_BLOCK = {v: k for k, v in BLOCK_MAP.items()}

@torch.no_grad()
def generate():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # --- 2. 加载全家桶权重 ---
    print(f"正在从 {MODEL_PATH} 加载模型...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # 初始化 UNet
    model = UNet2DConditionModel(
        sample_size=IMAGE_SIZE,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        cross_attention_dim=128,
    )
    model.load_state_dict(checkpoint['unet'])
    model.to(DEVICE).eval()

    # 初始化 Embedding 层
    label_emb = nn.Embedding(6, 128)
    label_emb.load_state_dict(checkpoint['emb'])
    label_emb.to(DEVICE).eval()

    # 初始化调度器
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # --- 3. 设定测试任务 (包含泛化性测试) ---
    # 格式: (左边方块, 右边方块)
    test_tasks = [
        ("diamond_ore", "none"),          # 单方块
        ("emerald_block", "none"), # 见过组合
        ("cobblestone", "none"),   # 泛化测试1 (未见过)
        ("brick", "none"),          # 泛化测试2 (未见过)
        ("planks_oak", "none"),       # 泛化测试3 (未见过)
    ]

    print(f"开始生成图片，采用 50 步快速采样...")

    for left_name, right_name in test_tasks:
        print(f"正在生成: [左: {left_name}] + [右: {right_name}]")
        
        # 准备语义 Embedding
        labels = torch.tensor([[BLOCK_MAP[left_name], BLOCK_MAP[right_name]]], device=DEVICE)
        encoder_hidden_states = label_emb(labels) # [1, 2, 128]

        # 准备初始噪声
        image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
        
        # 设置采样步数为 50 (加速推理)
        noise_scheduler.set_timesteps(50)

        # 采样循环
        for t in tqdm(noise_scheduler.timesteps, leave=False):
            # 预测噪声残差
            noise_pred = model(image, t, encoder_hidden_states=encoder_hidden_states).sample
            
            # 计算前一步的图像
            image = noise_scheduler.step(noise_pred, t, image).prev_sample

        # --- 4. 后处理并保存 ---
        image = (image / 2 + 0.5).clamp(0, 1) # 还原到 [0, 1]
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
        
        save_name = f"{left_name}_and_{right_name}.png"
        Image.fromarray(image).save(os.path.join(OUTPUT_DIR, save_name))
        print(f"保存成功: {save_name}")

    print(f"\n全部生成完毕！请查看目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate()