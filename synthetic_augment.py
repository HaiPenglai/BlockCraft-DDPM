import os
import random
import numpy as np
import math
from PIL import Image, ImageEnhance

# --- 配置 ---
TEXTURE_DIR = "block_textures"
OUTPUT_DIR = "dataset"
IMAGE_SIZE = 128
SAMPLES_PER_BLOCK = 100

SELECTED_BLOCKS = [
    "diamond_ore.png", # 钻石矿
    "emerald_block.png", # 绿宝石块
    "brick.png", # 红砖块
    "cobblestone.png", # 圆石
    "planks_oak.png", # 橡木板
    "stone.png",            # 石头
    "dirt.png",             # 泥土
    "sand.png",             # 沙子
    "gravel.png",           # 沙砾
    "bedrock.png",          # 基岩
    "iron_ore.png",         # 铁矿石
    "gold_ore.png",         # 金矿石
    "coal_ore.png",         # 煤矿石
    "lapis_block.png",      # 青金石块
    "obsidian.png",         # 黑曜石
    "glass.png",            # 玻璃 (透明特征学习)
    "stonebrick.png",       # 石砖
    "netherrack.png",       # 下界岩
    "bookshelf.png",        # 书架 (纹理较复杂，适合挑战模型)
    "glowstone.png",        # 萤石 (高亮度和复杂纹理)
]

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.array(matrix)
    B = np.array(pb).reshape(8)
    res = np.linalg.solve(A, B)
    return res

def project_3d_to_2d(points_3d, yaw, pitch, scale, cx, cy):
    projected_points = []
    for x, y, z in points_3d:
        # 1. 绕 Y 轴旋转 (Yaw)
        x_rot = x * math.cos(yaw) - z * math.sin(yaw)
        z_rot = x * math.sin(yaw) + z * math.cos(yaw)
        
        # 2. 绕 X 轴投影 (Pitch)
        y_final = y * math.cos(pitch) - z_rot * math.sin(pitch)
        x_final = x_rot
        
        # 3. 缩放
        canvas_x = cx + x_final * scale
        canvas_y = cy - y_final * scale 
        projected_points.append((canvas_x, canvas_y))
    return projected_points

def render_mc_block(texture_path, output_size=128):
    try:
        tex = Image.open(texture_path).convert("RGBA")
    except: return None
    
    s_tex = tex.width
    src_pts = [(0, 0), (0, s_tex), (s_tex, s_tex), (s_tex, 0)]

    # --- 随机与视角参数 ---
    # 基础 yaw 设为 45 度 (math.pi/4)，确保正对尖角
    # 在 45 度基础上左右旋转约 30 度
    yaw = math.pi/4 + math.radians(random.uniform(-30, 30))
    # 俯仰角，控制方块“压扁”程度，30-35度是MC的标准感
    pitch = math.radians(random.uniform(30, 35))
    # 缩小 scale，从之前的 50 降到 32-38，保证四周有黑边
    scale = random.uniform(32, 38) 
    
    cx, cy = output_size // 2, output_size // 2

    # 定义立方体顶点 (标准单位立方体)
    v3d = {
        't_front': (1, 1, 1),  't_left': (-1, 1, 1), 't_back': (-1, 1, -1), 't_right': (1, 1, -1),
        'b_front': (1, -1, 1), 'b_left': (-1, -1, 1), 'b_back': (-1, -1, -1), 'b_right': (1, -1, -1)
    }

    # 执行投影
    pts_2d_list = project_3d_to_2d(v3d.values(), yaw, pitch, scale, cx, cy)
    v2d = dict(zip(v3d.keys(), pts_2d_list))

    # --- 自动居中修正 ---
    all_x = [p[0] for p in pts_2d_list]
    all_y = [p[1] for p in pts_2d_list]
    center_x = (min(all_x) + max(all_x)) / 2
    center_y = (min(all_y) + max(all_y)) / 2
    offset_x = (output_size / 2) - center_x
    offset_y = (output_size / 2) - center_y
    for k in v2d:
        v2d[k] = (v2d[k][0] + offset_x, v2d[k][1] + offset_y)

    canvas = Image.new("RGBA", (output_size, output_size), (0, 0, 0, 255))

    # --- 渲染逻辑 (确保三个面完美拼接) ---
    
    # 1. 渲染左前侧面 (可见面 1)
    # 顶点顺序：左上, 左下, 中下, 中上
    left_f_dest = [v2d['t_left'], v2d['b_left'], v2d['b_front'], v2d['t_front']]
    l_tex = ImageEnhance.Brightness(tex).enhance(0.8 + random.uniform(-0.05, 0.05))
    coeffs = find_coeffs(left_f_dest, src_pts)
    face = l_tex.transform((output_size, output_size), Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    canvas.paste(face, (0,0), face)

    # 2. 渲染右前侧面 (可见面 2)
    # 顶点顺序：中上, 中下, 右下, 右上
    right_f_dest = [v2d['t_front'], v2d['b_front'], v2d['b_right'], v2d['t_right']]
    r_tex = ImageEnhance.Brightness(tex).enhance(0.6 + random.uniform(-0.05, 0.05))
    coeffs = find_coeffs(right_f_dest, src_pts)
    face = r_tex.transform((output_size, output_size), Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    canvas.paste(face, (0,0), face)

    # 3. 渲染顶面 (可见面 3)
    # 顶点顺序：后, 左, 前, 右
    top_dest = [v2d['t_back'], v2d['t_left'], v2d['t_front'], v2d['t_right']]
    t_tex = ImageEnhance.Brightness(tex).enhance(1.0 + random.uniform(-0.02, 0.02))
    coeffs = find_coeffs(top_dest, src_pts)
    face = t_tex.transform((output_size, output_size), Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    canvas.paste(face, (0,0), face)

    return canvas.convert("RGB")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    for block_file in SELECTED_BLOCKS:
        block_name = block_file.split('.')[0]
        out_path = os.path.join(OUTPUT_DIR, block_name)
        if not os.path.exists(out_path): os.makedirs(out_path)
        print(f"正在生成高质量且对齐的 {block_name} 数据集...")
        for i in range(SAMPLES_PER_BLOCK):
            img = render_mc_block(os.path.join(TEXTURE_DIR, block_file))
            if img:
                img.save(os.path.join(out_path, f"{block_name}_{i:04d}.png"))
    print("全部修复完毕！现在的方块大小适中且不再缺失侧面。")

if __name__ == "__main__":
    main()