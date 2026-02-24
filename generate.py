# --- START OF FILE generate.py ---
import os
import json
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler
from PIL import Image
from tqdm.auto import tqdm
import numpy as np

# --- 1. 配置参数 ---
# 指向你想读取的 epoch 模型路径 (比如 epoch_200 效果最好，就改成 200)
EPOCH_TO_LOAD = 200
MODEL_PATH = f"mc_blocks_ddpm_cond/epoch_{EPOCH_TO_LOAD:03d}/model.pt" 
MAPPING_PATH = "mc_blocks_ddpm_cond/class_mapping.json"
OUTPUT_DIR = "final_results"

IMAGE_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GUIDANCE_SCALE = 3.5 # CFG 引导尺度 (通常 3.0 ~ 7.0 效果最佳，越大越贴合条件)
INFERENCE_STEPS = 50 # 采样步数

@torch.no_grad()
def generate():
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)

    # 1. 加载类别映射表
    if not os.path.exists(MAPPING_PATH):
        raise FileNotFoundError(f"找不到类别映射文件: {MAPPING_PATH}。请确保训练已正常启动过。")
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        class_to_id = json.load(f)
    print("支持的方块种类:", list(class_to_id.keys()))

    # 2. 加载权重
    print(f"\n⏳ 正在从 {MODEL_PATH} 加载模型...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}。")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    num_classes = checkpoint['num_classes']
    
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

    # 初始化 Embedding 层 (+1 是为了包含 CFG 的无条件 Token)
    label_emb = nn.Embedding(num_classes + 1, 128)
    label_emb.load_state_dict(checkpoint['emb'])
    label_emb.to(DEVICE).eval()

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 3. 指定你想生成的方块列表
    # 你可以在这里修改你想生成的方块，比如我要生成 1个钻石、1个红砖块
    tasks = [
        "diamond_ore", # 钻石矿
        "emerald_block", # 绿宝石块
        "brick", # 红砖块
        "cobblestone", # 圆石
        "planks_oak", # 橡木板
        "stone",            # 石头
        "dirt",             # 泥土
        "sand",             # 沙子
        "gravel",           # 沙砾
        "bedrock",          # 基岩
        "iron_ore",         # 铁矿石
        "gold_ore",         # 金矿石
        "coal_ore",         # 煤矿石
        "lapis_block",      # 青金石块
        "obsidian",         # 黑曜石
        "glass",            # 玻璃 (透明特征学习)
        "stonebrick",       # 石砖
        "netherrack",       # 下界岩
        "bookshelf",        # 书架 (纹理较复杂，适合挑战模型)
        "glowstone",        # 萤石 (高亮度和复杂纹理)
    ]

    print(f"🚀 开始生成图片 (CFG Scale: {GUIDANCE_SCALE}, 步数: {INFERENCE_STEPS})")
    noise_scheduler.set_timesteps(INFERENCE_STEPS)

    for block_name in tasks:
        if block_name not in class_to_id:
            print(f"⚠️ 警告: 类别 '{block_name}' 不在训练集类别中，已跳过。")
            continue
            
        print(f"正在生成: {block_name} ...")
        
        # 准备 CFG 的条件 Embedding 和无条件 Embedding
        target_id = class_to_id[block_name]
        cond_id = torch.tensor([target_id], device=DEVICE)
        uncond_id = torch.tensor([num_classes], device=DEVICE) # 使用第 num_classes 作为空标签
        
        cond_emb = label_emb(cond_id).unsqueeze(1)    # [1, 1, 128]
        uncond_emb = label_emb(uncond_id).unsqueeze(1) # [1, 1, 128]

        # 初始随机噪声
        image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

        # 采样循环
        for t in tqdm(noise_scheduler.timesteps, leave=False):
            # 将 latent 和 emb 均复制成 2 份，前半部分给 unconditional，后半部分给 conditional
            latent_model_input = torch.cat([image] * 2)
            emb_input = torch.cat([uncond_emb, cond_emb])
            
            # 预测噪声
            noise_pred = model(latent_model_input, t, encoder_hidden_states=emb_input).sample
            
            # 执行 CFG (无分类器引导)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
            
            # 步进
            image = noise_scheduler.step(noise_pred_cfg, t, image).prev_sample

        # 4. 后处理并保存
        image = (image / 2 + 0.5).clamp(0, 1) # 还原到 [0, 1]
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
        
        save_name = f"result_{block_name}.png"
        Image.fromarray(image).save(os.path.join(OUTPUT_DIR, save_name))
        print(f"✅ 保存成功: {OUTPUT_DIR}/{save_name}")

    print("\n🎉 全部生成完毕！")

if __name__ == "__main__":
    generate()
# --- END OF FILE generate.py ---