import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
import matplotlib as mpl
import zlib
from math import sqrt

# ===================== 基础配置 =====================
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False
VIDEO_PATH = "godfather_clip.mp4"
FRAME_DIR = "frames_output"
FPS = 5
THRESHOLD = 0.06  # 镜头切分阈值
SMOOTH_WINDOW = 3
BASE_WINDOW_LEN = 10
COMP_WINDOW_LEN = 10
STEP = 3  # 滑动步长/短镜头合并阈值

# JPEG量化表（亮度量化表，简化处理）
JPEG_LUMA_QUANT_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

# Z字形扫描顺序
ZIGZAG_ORDER = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
]

def extract_frames():
    os.makedirs(FRAME_DIR, exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", VIDEO_PATH,
        "-r", str(FPS),
        "-q:v", "2",
        os.path.join(FRAME_DIR, "frame_%04d.jpg")
    ]
    try:
        result = subprocess.run(
            ffmpeg_cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print("=== 视频帧提取完成 ===")
        return True
    except subprocess.CalledProcessError as e:
        print(f"=== 帧提取失败 ===")
        print(f"错误详情：{e.stderr.strip()}")
        exit(1)


def show_frame(frame_name, title):
    frame_path = os.path.join(FRAME_DIR, frame_name)
    img = Image.open(frame_path).convert('RGB')
    img_array = np.array(img)
    pixel_mean = np.mean(img_array)

    plt.figure(figsize=(8, 6))
    plt.imshow(img_array)
    plt.title(f"{title}（像素均值：{pixel_mean:.2f}）", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"【{title}】")
    print(f"  文件名称：{frame_name}")
    print(f"  像素均值：{pixel_mean:.2f}\n")


def print_and_show_extract_result():
    frame_files = sorted([
        f for f in os.listdir(FRAME_DIR) if f.startswith("frame_") and f.endswith(".jpg")
    ])
    if not frame_files:
        print("=== 提取结果异常 ===")
        print("未检测到任何提取的帧文件！")
        return

    total_frames = len(frame_files)
    video_duration = total_frames / FPS

    print("=== 帧提取结果统计 ===")
    print(f"总提取帧数：{total_frames} 帧")
    print(f"提取帧率：{FPS} fps")
    print(f"估算视频时长：{video_duration:.2f} 秒")
    print(f"帧文件范围：{frame_files[0]} ~ {frame_files[-1]}\n")

    first_frame = frame_files[0]
    middle_frame = frame_files[total_frames // 2]
    last_frame = frame_files[-1]

    print("=== 展示关键帧 ===")
    show_frame(first_frame, "首帧")
    show_frame(middle_frame, "中间帧")
    show_frame(last_frame, "末尾帧")


def calc_frame_hist(frame_path):
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"警告：无法读取帧文件 {frame_path}，返回空直方图")
        return np.zeros((18 * 8 * 8 + 16,))

    frame = cv2.resize(frame, (320, 240))
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycbcr)

    hist_y = cv2.calcHist([y], [0], None, [32], [0, 256])
    hist_cr = cv2.calcHist([cr], [0], None, [16], [0, 256])
    hist_cb = cv2.calcHist([cb], [0], None, [16], [0, 256])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    hist_grad = cv2.calcHist([grad_mag.astype(np.uint8)], [0], None, [16], [0, 256])

    hist_y = cv2.normalize(hist_y, hist_y).flatten()
    hist_cr = cv2.normalize(hist_cr, hist_cr).flatten()
    hist_cb = cv2.normalize(hist_cb, hist_cb).flatten()
    hist_grad = cv2.normalize(hist_grad, hist_grad).flatten()

    hist = np.concatenate([
        hist_y * 0.4,
        hist_cr * 0.2,
        hist_cb * 0.2,
        hist_grad * 0.2
    ])
    return hist


def calc_hist_diff():
    frame_files = sorted([
        f for f in os.listdir(FRAME_DIR) if f.startswith("frame_") and f.endswith(".jpg")
    ])
    if len(frame_files) < 2:
        print("帧数量不足，无法计算差值！")
        exit(1)

    hist_list = []
    frame_times = []
    frame_indices = []

    for frame_name in frame_files:
        hist = calc_frame_hist(os.path.join(FRAME_DIR, frame_name))
        hist_list.append(hist)
        try:
            frame_idx = int(frame_name.split("_")[1].split(".")[0])
            frame_times.append(frame_idx / FPS)
            frame_indices.append(frame_idx)
        except (IndexError, ValueError):
            print(f"警告：帧文件命名异常 {frame_name}，跳过时间计算")
            frame_times.append(len(frame_times) / FPS)
            frame_indices.append(len(frame_indices) + 1)

    hist_diff = []
    for i in range(1, len(hist_list)):
        diff = cv2.compareHist(hist_list[i - 1], hist_list[i], cv2.HISTCMP_BHATTACHARYYA)
        hist_diff.append(diff)

    hist_diff = np.array(hist_diff)
    if np.max(hist_diff) > 0:
        hist_diff = (hist_diff - np.min(hist_diff)) / (np.max(hist_diff) - np.min(hist_diff))

    if len(hist_diff) >= SMOOTH_WINDOW:
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        hist_diff = np.convolve(hist_diff, kernel, mode='same')

    diff_deriv = np.gradient(hist_diff)
    hist_diff = hist_diff * 0.5 + np.abs(diff_deriv) * 0.5

    return hist_diff, frame_times, frame_files, frame_indices


def split_shots_by_frames(hist_diff, frame_files, frame_indices, threshold):
    shots = []
    shot_id = 1
    start_frame_idx = int(frame_files[0].split("_")[1].split(".")[0])
    boundary_indices = []

    if len(hist_diff) < BASE_WINDOW_LEN:
        print("帧数量不足，无法计算基准窗口均值！")
        shots.append({
            "镜头ID": shot_id,
            "帧范围": [start_frame_idx, int(frame_files[-1].split("_")[1].split(".")[0])],
            "时间范围": [0.0, round(len(frame_files) / FPS, 1)],
            "帧文件范围": f"{frame_files[0]} ~ {frame_files[-1]}"
        })
        return shots

    base_window = hist_diff[:BASE_WINDOW_LEN]
    base_mean = np.mean(base_window)

    comp_start = 3
    while comp_start + COMP_WINDOW_LEN <= len(hist_diff):
        comp_window = hist_diff[comp_start:comp_start + COMP_WINDOW_LEN]
        comp_mean = np.mean(comp_window)
        mean_diff = comp_mean - base_mean

        if abs(mean_diff) > threshold:
            bound_idx = comp_start + COMP_WINDOW_LEN // 2
            boundary_indices.append(bound_idx)

            end_frame_idx = frame_indices[bound_idx] - 1
            start_time = start_frame_idx / FPS
            end_time = end_frame_idx / FPS

            shots.append({
                "镜头ID": shot_id,
                "帧范围": [start_frame_idx, end_frame_idx],
                "时间范围": [round(start_time, 1), round(end_time, 1)],
                "帧文件范围": f"{frame_files[start_frame_idx - 1]} ~ {frame_files[end_frame_idx - 1]}"
            })

            start_frame_idx = frame_indices[bound_idx]
            shot_id += 1

        comp_start += STEP

    last_frame_idx = int(frame_files[-1].split("_")[1].split(".")[0])
    start_time = start_frame_idx / FPS
    end_time = last_frame_idx / FPS
    shots.append({
        "镜头ID": shot_id,
        "帧范围": [start_frame_idx, last_frame_idx],
        "时间范围": [round(start_time, 1), round(end_time, 1)],
        "帧文件范围": f"{frame_files[start_frame_idx - 1]} ~ {frame_files[-1]}"
    })

    if len(shots) <= 1:
        return shots

    merged_shots = []
    prev_shot = shots[0]
    for shot in shots[1:]:
        curr_shot_frames = shot["帧范围"][1] - shot["帧范围"][0] + 1
        if curr_shot_frames <= STEP:
            merged_shot = {
                "镜头ID": prev_shot["镜头ID"],
                "帧范围": [prev_shot["帧范围"][0], shot["帧范围"][1]],
                "时间范围": [prev_shot["时间范围"][0], shot["时间范围"][1]],
                "帧文件范围": f"{prev_shot['帧文件范围'].split(' ~ ')[0]} ~ {shot['帧文件范围'].split(' ~ ')[1]}"
            }
            prev_shot = merged_shot
        else:
            merged_shots.append(prev_shot)
            prev_shot = shot
    merged_shots.append(prev_shot)

    for idx, merged_shot in enumerate(merged_shots):
        merged_shot["镜头ID"] = idx + 1

    print(f"\n=== 短镜头合并结果 ===")
    print(f"合并前镜头数：{len(shots)} | 合并后镜头数：{len(merged_shots)}")
    return merged_shots


def print_shot_result(shots):
    print("\n" + "=" * 60)
    print("镜头切分结果（滑动窗口 + 短镜头合并）")
    print("=" * 60)
    for shot in shots:
        shot_frames = shot["帧范围"][1] - shot["帧范围"][0] + 1
        print(
            f"{shot['镜头ID']}号镜头：第{shot['帧范围'][0]}-{shot['帧范围'][1]}帧（共{shot_frames}帧）| 对应视频{shot['时间范围'][0]}-{shot['时间范围'][1]}秒")
        print(f"  对应帧文件：{shot['帧文件范围']}")


def detect_shot_boundary(hist_diff, frame_times, frame_files, frame_indices, shots):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(hist_diff))
    ax.bar(x, hist_diff, color='skyblue', label='增强型帧间差值（归一化+平滑）')
    ax.axhline(y=THRESHOLD, color='red', linestyle='--', label=f'阈值 = {THRESHOLD}')

    merged_boundary_indices = []
    for shot in shots[:-1]:
        end_frame = shot["帧范围"][1]
        if end_frame in frame_indices:
            diff_idx = frame_indices.index(end_frame)
            merged_boundary_indices.append(diff_idx)

    for i, bound_idx in enumerate(merged_boundary_indices):
        ax.axvline(x=bound_idx, color='darkred', linestyle='-', linewidth=2,
                   label='合并后镜头切分边界' if i == 0 else "")
        ax.text(bound_idx, np.max(hist_diff) * 0.9, f'边界{i + 1}',
                ha='center', va='center', color='darkred', fontweight='bold')

    ax.set_xlabel('相邻帧对（如 1→2 表示第1帧与第2帧的差值）')
    ax.set_ylabel('归一化差值（越大差异越明显）')
    ax.set_title('视频帧间增强型差值分布（含合并后镜头切分边界）')
    ax.legend()

    step = max(1, len(hist_diff) // 10)
    xticks_pos = x[::step]
    xticks_labels = [f"{i + 1}→{i + 2}\n({frame_times[i]:.1f}s)" for i in xticks_pos]
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_labels, rotation=0)

    plt.tight_layout()
    hist_img_path = os.path.join(FRAME_DIR, "hist_diff_bar_optimized.png")
    plt.savefig(hist_img_path)
    print(f"优化后差值柱状图已保存至：{hist_img_path}")
    plt.show()

    shot_boundaries = []
    if len(hist_diff) >= BASE_WINDOW_LEN:
        base_mean = np.mean(hist_diff[:BASE_WINDOW_LEN])
        comp_start = 3
        while comp_start + COMP_WINDOW_LEN <= len(hist_diff):
            comp_mean = np.mean(hist_diff[comp_start:comp_start + COMP_WINDOW_LEN])
            mean_diff = comp_mean - base_mean
            if mean_diff > THRESHOLD:
                bound_idx = comp_start + COMP_WINDOW_LEN // 2
                boundary_time = frame_times[bound_idx]
                boundary_frame_pair = f"{bound_idx + 1}→{bound_idx + 2}"
                shot_boundaries.append({
                    "帧对": boundary_frame_pair,
                    "视频时间（秒）": round(boundary_time, 1),
                    "均值差": round(mean_diff, 3),
                    "差值": round(hist_diff[bound_idx], 3)
                })
            comp_start += STEP

    print("\n检测到的原始镜头边界（合并前）：")
    if not shot_boundaries:
        print("未检测到超过阈值的镜头边界！")
    else:
        for idx, boundary in enumerate(shot_boundaries):
            print(
                f"边界{idx + 1}：帧对{boundary['帧对']} | 时间{boundary['视频时间（秒）']}s | 均值差{boundary['均值差']} | 差值{boundary['差值']}")
    return shot_boundaries



def jpeg_compress_i_frame(frame_path, frame_name="I帧"):
    """
    对单张I帧图像执行JPEG压缩（简化版）
    步骤：1.RGB转YUV 2.8x8分块 3.DCT变换 4.量化 5.游程编码 6.zlib压缩 7.计算压缩率
    """
    # 1. 读取图像并转换为RGB
    img = Image.open(frame_path).convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    h, w, _ = img_array.shape
    print(f"\n=== 开始{frame_name} JPEG压缩 ===")
    print(f"原始图像尺寸：{w}x{h}，原始RGB数据大小：{img_array.nbytes / 1024:.2f} KB")

    # 2. RGB转YUV（仅处理Y分量，简化计算）
    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B  # 亮度分量
    Y = Y - 128  # DCT变换前中心化（0均值）
    print(f"Y分量转换完成，Y分量均值：{np.mean(Y):.2f}")

    # 3. 补全图像为8的倍数（避免分块越界）
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    Y_padded = np.pad(Y, ((0, pad_h), (0, pad_w)), mode='constant')
    h_pad, w_pad = Y_padded.shape
    print(f"补全后Y分量尺寸：{w_pad}x{h_pad}")

    # 4. 8x8分块 + DCT变换 + 量化
    dct_quant_blocks = []
    for i in range(0, h_pad, 8):
        for j in range(0, w_pad, 8):
            # 提取8x8块
            block = Y_padded[i:i + 8, j:j + 8]
            # DCT变换（cv2.dct要求float32）
            dct_block = cv2.dct(block)
            # 量化（除以量化表）
            quant_block = np.round(dct_block / JPEG_LUMA_QUANT_TABLE)
            dct_quant_blocks.append(quant_block)

    # 输出前2个块的量化结果（中间数据）
    print(f"\n前2个8x8块的DCT量化结果（示例）：")
    print(f"块1：\n{dct_quant_blocks[0].astype(np.int16)}")
    print(f"块2：\n{dct_quant_blocks[1].astype(np.int16)}")

    # 5. Z字形扫描 + 游程编码
    run_length_data = []
    for block in dct_quant_blocks:
        # Z字形扫描
        zigzag_block = block.flatten()[ZIGZAG_ORDER]
        # 游程编码（简化：(0的个数, 非0值)）
        zero_count = 0
        for val in zigzag_block:
            if val == 0:
                zero_count += 1
            else:
                run_length_data.append((zero_count, val))
                zero_count = 0
        # 结束标记
        run_length_data.append((-1, -1))

    # 输出前10个游程编码结果（中间数据）
    print(f"\n前10个游程编码结果（示例）：{run_length_data[:10]}")

    # 6. zlib压缩（模拟JPEG的熵编码）
    # 将游程数据转为字节流
    byte_data = b''
    for (zero, val) in run_length_data:
        # 转换为Python原生int，避免numpy类型无to_bytes方法
        zero_int = int(zero)
        val_int = int(val)
        # 关键：zero_int可能为-1（结束标记），必须指定signed=True
        byte_data += zero_int.to_bytes(2, byteorder='little', signed=True) + val_int.to_bytes(2, byteorder='little',signed=True)
    compressed_data = zlib.compress(byte_data)
    compressed_size = len(compressed_data) / 1024  # KB
    original_size = img_array.nbytes / 1024  # KB
    compression_ratio = original_size / compressed_size

    # 7. 输出压缩结果
    print(f"\n=== {frame_name} JPEG压缩结果 ===")
    print(f"原始RGB数据大小：{original_size:.2f} KB")
    print(f"压缩后数据大小：{compressed_size:.2f} KB")
    print(f"压缩率：{compression_ratio:.2f} : 1")

    return {
        "frame_name": frame_name,
        "original_size_kb": original_size,
        "compressed_size_kb": compressed_size,
        "compression_ratio": compression_ratio
    }

def mpeg_compress_p_frame(i_frame_path, p_frame_path):
    """
    P帧压缩（基于参考I帧的块匹配）
    步骤：1.读取I/P帧 2.8x8块匹配（48x64搜索范围） 3.差值计算 4.差值编码 5.计算压缩率
    """
    # 1. 读取参考I帧和P帧（仅处理Y分量）
    i_img = Image.open(i_frame_path).convert('RGB')
    p_img = Image.open(p_frame_path).convert('RGB')
    i_array = np.array(i_img, dtype=np.float32)
    p_array = np.array(p_img, dtype=np.float32)
    h, w, _ = i_array.shape

    # 转Y分量
    i_Y = 0.299 * i_array[:, :, 0] + 0.587 * i_array[:, :, 1] + 0.114 * i_array[:, :, 2] - 128
    p_Y = 0.299 * p_array[:, :, 0] + 0.587 * p_array[:, :, 1] + 0.114 * p_array[:, :, 2] - 128

    print(f"\n=== 开始P帧压缩（参考I帧：{os.path.basename(i_frame_path)}）===")
    print(f"参考I帧/P帧尺寸：{w}x{h}")
    print(f"I帧Y分量均值：{np.mean(i_Y):.2f}，P帧Y分量均值：{np.mean(p_Y):.2f}")

    # 2. 补全为8的倍数
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    i_Y_pad = np.pad(i_Y, ((0, pad_h), (0, pad_w)), mode='constant')
    p_Y_pad = np.pad(p_Y, ((0, pad_h), (0, pad_w)), mode='constant')
    h_pad, w_pad = i_Y_pad.shape

    # 3. 块匹配（每个P帧块在I帧对应位置48x64范围找最佳匹配）
    best_matches = []  # 存储每个块的最佳匹配位置
    diff_blocks = []  # 存储每个块的差值
    search_range_x = 24  # 48范围：±24
    search_range_y = 32  # 64范围：±32

    for i in range(0, h_pad, 8):
        for j in range(0, w_pad, 8):
            # 当前P帧块
            p_block = p_Y_pad[i:i + 8, j:j + 8]
            # 搜索范围：x∈[j-24, j+24], y∈[i-32, i+32]（边界限制）
            start_x = max(0, j - search_range_x)
            end_x = min(w_pad - 8, j + search_range_x)
            start_y = max(0, i - search_range_y)
            end_y = min(h_pad - 8, i + search_range_y)

            # 计算所有候选块的MSE，找最小值
            min_mse = float('inf')
            best_pos = (j, i)
            for y in range(start_y, end_y + 1, 8):
                for x in range(start_x, end_x + 1, 8):
                    i_block = i_Y_pad[y:y + 8, x:x + 8]
                    mse = np.mean((p_block - i_block) ** 2)
                    if mse < min_mse:
                        min_mse = mse
                        best_pos = (x, y)

            # 最佳匹配块
            best_i_block = i_Y_pad[best_pos[1]:best_pos[1] + 8, best_pos[0]:best_pos[0] + 8]
            # 计算差值块
            diff_block = p_block - best_i_block
            best_matches.append(best_pos)
            diff_blocks.append(diff_block)

    # 输出前2个块的匹配结果（中间数据）
    print(f"\n前2个P帧块的匹配结果（示例）：")
    print(
        f"块1（位置(0,0)）：最佳匹配位置{best_matches[0]}，MSE={np.mean((p_Y_pad[0:8, 0:8] - i_Y_pad[best_matches[0][1]:best_matches[0][1] + 8, best_matches[0][0]:best_matches[0][0] + 8]) ** 2):.2f}")
    print(
        f"块2（位置(0,8)）：最佳匹配位置{best_matches[1]}，MSE={np.mean((p_Y_pad[0:8, 8:16] - i_Y_pad[best_matches[1][1]:best_matches[1][1] + 8, best_matches[1][0]:best_matches[1][0] + 8]) ** 2):.2f}")
    print(f"块1差值（示例）：\n{diff_blocks[0].astype(np.int16)}")

    # 4. 差值块编码（复用I帧的JPEG编码流程）
    # 量化差值块
    quant_diff_blocks = [np.round(block / JPEG_LUMA_QUANT_TABLE) for block in diff_blocks]
    # 游程编码
    run_length_diff = []
    for block in quant_diff_blocks:
        zigzag_block = block.flatten()[ZIGZAG_ORDER]
        zero_count = 0
        for val in zigzag_block:
            if val == 0:
                zero_count += 1
            else:
                run_length_diff.append((zero_count, val))
                zero_count = 0
        run_length_diff.append((-1, -1))

    # 5. zlib压缩
    # P帧原始数据大小（Y分量）
    p_original_size = p_array.nbytes / 1024
    # 差值数据压缩
    byte_diff = b''
    for (zero, val) in run_length_diff:
        # 转换numpy类型为Python原生int
        zero_int = int(zero)
        val_int = int(val)
        # 对zero_int也指定signed=True，兼容-1结束标记
        byte_diff += zero_int.to_bytes(2, byteorder='little', signed=True) + val_int.to_bytes(2, byteorder='little',                                                                              signed=True)
    byte_pos = b''
    for (x, y) in best_matches:
        byte_pos += x.to_bytes(2, byteorder='little') + y.to_bytes(2, byteorder='little')
    # 总压缩数据
    p_compressed_data = zlib.compress(byte_diff + byte_pos)
    p_compressed_size = len(p_compressed_data) / 1024
    p_compression_ratio = p_original_size / p_compressed_size

    # 6. 输出P帧压缩结果
    print(f"\n=== P帧压缩结果 ===")
    print(f"P帧原始RGB数据大小：{p_original_size:.2f} KB")
    print(f"P帧压缩后数据大小（差值+匹配位置）：{p_compressed_size:.2f} KB")
    print(f"P帧压缩率：{p_compression_ratio:.2f} : 1")

    # 补充I帧压缩率（对比）
    i_compress_result = jpeg_compress_i_frame(i_frame_path, "参考I帧")
    print(f"\n=== I/P帧压缩率对比 ===")
    print(f"参考I帧压缩率：{i_compress_result['compression_ratio']:.2f} : 1")
    print(f"P帧压缩率：{p_compression_ratio:.2f} : 1")

    return {
        "p_original_size_kb": p_original_size,
        "p_compressed_size_kb": p_compressed_size,
        "p_compression_ratio": p_compression_ratio,
        "i_compression_ratio": i_compress_result['compression_ratio']
    }


if __name__ == "__main__":
    # 1. 提取帧 + 镜头切分
    extract_success = extract_frames()
    if extract_success:
        print_and_show_extract_result()
    hist_diff, frame_times, frame_files, frame_indices = calc_hist_diff()
    shots = split_shots_by_frames(hist_diff, frame_files, frame_indices, THRESHOLD)
    shot_boundaries = detect_shot_boundary(hist_diff, frame_times, frame_files, frame_indices, shots)
    print_shot_result(shots)

    # 2. 选择第一个镜头作为实验数据
    if len(shots) == 0:
        print("无有效镜头，退出压缩实验")
        exit(1)
    target_shot = shots[0]
    print(f"\n=== 选择{target_shot['镜头ID']}号镜头进行MPEG压缩实验 ===")
    print(f"镜头帧范围：{target_shot['帧范围'][0]}-{target_shot['帧范围'][1]}帧")

    # 提取镜头内的帧文件列表
    shot_frame_files = [f for f in frame_files if
                        int(f.split("_")[1].split(".")[0]) >= target_shot['帧范围'][0] and
                        int(f.split("_")[1].split(".")[0]) <= target_shot['帧范围'][1]]
    shot_frame_files.sort()
    if len(shot_frame_files) < 3:
        print("镜头帧数不足3帧，无法选择首/中/尾I帧")
        exit(1)

    # 3. 选择I帧（首帧、中间帧、末尾帧）
    i1_frame_file = shot_frame_files[0]  # 首帧I1
    i2_frame_file = shot_frame_files[len(shot_frame_files) // 2]  # 中间帧I2
    i3_frame_file = shot_frame_files[-1]  # 末尾帧I3
    i1_frame_path = os.path.join(FRAME_DIR, i1_frame_file)
    i2_frame_path = os.path.join(FRAME_DIR, i2_frame_file)
    i3_frame_path = os.path.join(FRAME_DIR, i3_frame_file)

    # 4. I帧JPEG压缩实验
    print("\n" + "=" * 80)
    print("==================== I帧JPEG压缩实验 ====================")
    i1_result = jpeg_compress_i_frame(i1_frame_path, f"首I帧（{i1_frame_file}）")
    i2_result = jpeg_compress_i_frame(i2_frame_path, f"中I帧（{i2_frame_file}）")
    i3_result = jpeg_compress_i_frame(i3_frame_path, f"尾I帧（{i3_frame_file}）")

    # 5. P帧压缩实验（选择中间I帧的下一帧作为P帧）
    print("\n" + "=" * 80)
    print("==================== P帧压缩实验 ====================")
    i2_idx = shot_frame_files.index(i2_frame_file)
    if i2_idx + 1 >= len(shot_frame_files):
        print("中间I帧已是镜头最后一帧，无P帧可选")
    else:
        p_frame_file = shot_frame_files[i2_idx + 1]
        p_frame_path = os.path.join(FRAME_DIR, p_frame_file)
        p_result = mpeg_compress_p_frame(i2_frame_path, p_frame_path)

    # 6. 汇总结果展示
    print("\n" + "=" * 80)
    print("==================== 压缩实验汇总结果 ====================")
    print(f"1. 首I帧压缩率：{i1_result['compression_ratio']:.2f} : 1")
    print(f"2. 中I帧压缩率：{i2_result['compression_ratio']:.2f} : 1")
    print(f"3. 尾I帧压缩率：{i3_result['compression_ratio']:.2f} : 1")
    if 'p_result' in locals():
        print(f"4. P帧（{p_frame_file}）压缩率：{p_result['p_compression_ratio']:.2f} : 1")
        print(f"   参考I帧压缩率：{p_result['i_compression_ratio']:.2f} : 1")