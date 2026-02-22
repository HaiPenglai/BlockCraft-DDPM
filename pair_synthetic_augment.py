import os
import random
import glob
from PIL import Image

# --- 配置 ---
SINGLE_DATASET_DIR = "dataset" 
OUTPUT_PAIR_DIR = "dataset_pairs"
IMAGE_SIZE = 128
BLOCK_SIZE = 64  # 将 128 缩小一半，确保左右各占 64 像素，互不重叠

BLOCKS = [
    "diamond_ore",
    "emerald_block",
    "brick",
    "cobblestone",
    "planks_oak",
]

# 泛化性排除列表（训练集不包含这些组合，用于测试模型是否理解语义）
EXCLUDED_PAIRS = [
    ("emerald_block", "planks_oak"),
    ("brick", "planks_oak"),
    ("diamond_ore", "cobblestone"),
    ("cobblestone", "diamond_ore"),    # 完全没有见过的钻石+圆石组合能否生成出来？
    ("emerald_block", "emerald_block")  # 没有见过的自己和自己组合，能否生成出来？
]

SAMPLES_PER_PAIR = 50

def create_pair_dataset():
    if not os.path.exists(OUTPUT_PAIR_DIR):
        os.makedirs(OUTPUT_PAIR_DIR)
    
    metadata = []
    print(f"正在生成『左右对齐、无重叠』的双方块数据集...")

    for left_name in BLOCKS:
        for right_name in BLOCKS:
            if (left_name, right_name) in EXCLUDED_PAIRS:
                print(f"跳过排除组合 (用于后期测试): {left_name} + {right_name}")
                continue
            
            pair_folder_name = f"{left_name}_and_{right_name}"
            save_path = os.path.join(OUTPUT_PAIR_DIR, pair_folder_name)
            if not os.path.exists(save_path): os.makedirs(save_path)

            left_images = glob.glob(os.path.join(SINGLE_DATASET_DIR, left_name, "*.png"))
            right_images = glob.glob(os.path.join(SINGLE_DATASET_DIR, right_name, "*.png"))

            samples = SAMPLES_PER_PAIR 
            
            for i in range(samples):
                img_l_path = random.choice(left_images)
                img_r_path = random.choice(right_images)
                
                # 1. 处理左侧方块
                # 将 128x128 的原图缩小到 64x64
                img_l = Image.open(img_l_path).convert("RGB")
                img_l = img_l.resize((BLOCK_SIZE, BLOCK_SIZE), Image.Resampling.LANCZOS)
                
                # 2. 处理右侧方块
                img_r = Image.open(img_r_path).convert("RGB")
                img_r = img_r.resize((BLOCK_SIZE, BLOCK_SIZE), Image.Resampling.LANCZOS)

                # 3. 创建 128x128 的全黑背景
                canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

                # 4. 粘贴方块
                # 高度居中计算: (128 - 64) // 2 = 32
                y_pos = (IMAGE_SIZE - BLOCK_SIZE) // 2 
                
                # 左方块顶着最左边粘贴 (x=0)
                canvas.paste(img_l, (0, y_pos))
                
                # 右方块顶着正中间粘贴 (x=64)
                canvas.paste(img_r, (BLOCK_SIZE, y_pos))

                # 保存图片
                file_name = f"{pair_folder_name}_{i:04d}.png"
                canvas.save(os.path.join(save_path, file_name))
                
                # 记录标签: 使用非常明确的方位词，方便 CLIP 学习
                prompt = f"a {left_name} on the left and a {right_name} on the right"
                metadata.append(f"{pair_folder_name}/{file_name}\t{prompt}")

    print(f"数据合成完毕！")
    print(f"每张图已划分为: 左侧(0-63px)放置{BLOCK_SIZE}px方块, 右侧(64-127px)放置{BLOCK_SIZE}px方块。")

if __name__ == "__main__":
    create_pair_dataset()