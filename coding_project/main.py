import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False
VIDEO_PATH = "godfather_clip.mp4"
FRAME_DIR = "frames_output"
FPS = 5
THRESHOLD = 0.24
SMOOTH_WINDOW = 3


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
        print(f"FFmpeg执行输出：{result.stdout.strip()}")
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

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 6))
    plt.imshow(img_array)
    plt.title(f"{title}（像素均值：{pixel_mean:.2f}）", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"【{title}】")
    print(f"  文件名称：{frame_name}")
    print(f"  文件路径：{frame_path}")
    print(f"  像素均值：{pixel_mean:.2f}\n")


def print_and_show_extract_result():
    frame_files = sorted([
        f for f in os.listdir(FRAME_DIR)
        if f.startswith("frame_") and f.endswith(".jpg")
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


# ===================== 核心改进1：增强直方图特征 =====================
def calc_frame_hist(frame_path):
    """
    改进点：
    1. 增加图像降采样，减少计算量同时降低噪声
    2. 分离亮度/色度通道，分别计算直方图后融合
    3. 增加梯度直方图补充纹理特征
    """
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"警告：无法读取帧文件 {frame_path}，返回空直方图")
        return np.zeros((18 * 8 * 8 + 16,))  # 预留梯度特征维度

    # 1. 降采样（缩小到320x240），降低噪声和计算量
    frame = cv2.resize(frame, (320, 240))
    # 2. 转换为YCbCr（视频标准空间，亮度/色度分离）
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycbcr)

    # 3. 分别计算亮度和色度直方图
    # 亮度直方图（Y通道，分32箱，更敏感）
    hist_y = cv2.calcHist([y], [0], None, [32], [0, 256])
    # 色度直方图（Cr/Cb通道，各16箱）
    hist_cr = cv2.calcHist([cr], [0], None, [16], [0, 256])
    hist_cb = cv2.calcHist([cb], [0], None, [16], [0, 256])

    # 4. 计算梯度直方图（补充纹理特征）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    hist_grad = cv2.calcHist([grad_mag.astype(np.uint8)], [0], None, [16], [0, 256])

    # 5. 归一化并融合所有特征
    hist_y = cv2.normalize(hist_y, hist_y).flatten()
    hist_cr = cv2.normalize(hist_cr, hist_cr).flatten()
    hist_cb = cv2.normalize(hist_cb, hist_cb).flatten()
    hist_grad = cv2.normalize(hist_grad, hist_grad).flatten()

    # 融合（亮度权重0.4，色度各0.2，梯度0.2）
    hist = np.concatenate([
        hist_y * 0.4,
        hist_cr * 0.2,
        hist_cb * 0.2,
        hist_grad * 0.2
    ])
    return hist


# ===================== 核心改进2：优化差值计算 =====================
def calc_hist_diff():
    """
    改进点：
    1. 使用巴氏距离（Bhattacharyya）替代卡方距离，对细微差异更敏感
    2. 差值归一化到[0,1]，统一阈值参考
    3. 滑动窗口平滑，过滤单帧噪声
    4. 计算帧间差值的相对变化率，增强边界识别
    """
    frame_files = sorted(
        [f for f in os.listdir(FRAME_DIR) if f.startswith("frame_") and f.endswith(".jpg")])
    if len(frame_files) < 2:
        print("帧数量不足，无法计算差值！")
        exit(1)

    hist_list = []  # 预存所有帧的直方图，避免重复计算
    frame_times = []
    frame_indices = []

    # 第一步：预计算所有帧的直方图
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

    # 第二步：计算帧间巴氏距离（对细微差异更敏感）
    hist_diff = []
    for i in range(1, len(hist_list)):
        # 巴氏距离（范围[0,1]，值越大差异越大）
        diff = cv2.compareHist(hist_list[i - 1], hist_list[i], cv2.HISTCMP_BHATTACHARYYA)
        hist_diff.append(diff)

    # 第三步：归一化差值到[0,1]
    hist_diff = np.array(hist_diff)
    if np.max(hist_diff) > 0:
        hist_diff = (hist_diff - np.min(hist_diff)) / (np.max(hist_diff) - np.min(hist_diff))

    # 第四步：滑动窗口平滑，过滤单帧噪声
    if len(hist_diff) >= SMOOTH_WINDOW:
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        hist_diff = np.convolve(hist_diff, kernel, mode='same')

    # 第五步：计算相对变化率（增强边界突变）
    diff_deriv = np.gradient(hist_diff)  # 差值的一阶导数（变化率）
    # 融合原始差值和变化率（权重各0.5）
    hist_diff = hist_diff * 0.5 + np.abs(diff_deriv) * 0.5

    return hist_diff, frame_times, frame_files, frame_indices


def split_shots_by_frames(hist_diff, frame_files, frame_indices, threshold):
    shots = []
    shot_id = 1
    start_frame_idx = int(frame_files[0].split("_")[1].split(".")[0])

    for i, diff in enumerate(hist_diff):
        if diff > threshold:
            end_frame_idx = frame_indices[i] - 1
            start_time = start_frame_idx / FPS
            end_time = end_frame_idx / FPS

            shots.append({
                "镜头ID": shot_id,
                "帧范围": [start_frame_idx, end_frame_idx],
                "时间范围": [round(start_time, 1), round(end_time, 1)],
                "帧文件范围": f"{frame_files[start_frame_idx - 1]} ~ {frame_files[end_frame_idx - 1]}"
            })

            start_frame_idx = frame_indices[i]
            shot_id += 1

    last_frame_idx = int(frame_files[-1].split("_")[1].split(".")[0])
    start_time = start_frame_idx / FPS
    end_time = last_frame_idx / FPS
    shots.append({
        "镜头ID": shot_id,
        "帧范围": [start_frame_idx, last_frame_idx],
        "时间范围": [round(start_time, 1), round(end_time, 1)],
        "帧文件范围": f"{frame_files[start_frame_idx - 1]} ~ {frame_files[-1]}"
    })

    return shots


def print_shot_result(shots):
    print("\n" + "=" * 60)
    print("📸 镜头切分结果（按帧/时间范围区分）")
    print("=" * 60)
    for shot in shots:
        print(
            f"{shot['镜头ID']}号镜头：第{shot['帧范围'][0]}-{shot['帧范围'][1]}帧（对应视频{shot['时间范围'][0]}-{shot['时间范围'][1]}秒）")
        print(f"  对应帧文件：{shot['帧文件范围']}")


def detect_shot_boundary(hist_diff, frame_times, frame_files, frame_indices):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(hist_diff))
    ax.bar(x, hist_diff, color='skyblue', label='增强型帧间差值（归一化+平滑）')
    ax.axhline(y=THRESHOLD, color='red', linestyle='--', label=f'阈值 = {THRESHOLD}')

    ax.set_xlabel('相邻帧对（如 1→2 表示第1帧与第2帧的差值）')
    ax.set_ylabel('归一化差值（越大差异越明显）')
    ax.set_title('视频帧间增强型差值分布（优化后）')
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
    for i, diff in enumerate(hist_diff):
        if diff > THRESHOLD:
            boundary_time = frame_times[i]
            boundary_frame_pair = f"{i + 1}→{i + 2}"
            shot_boundaries.append({
                "帧对": boundary_frame_pair,
                "视频时间（秒）": round(boundary_time, 1),
                "差值": round(diff, 3)
            })

    print("\n检测到的镜头边界（优化后）：")
    if not shot_boundaries:
        print("未检测到超过阈值的镜头边界！")
    else:
        for idx, boundary in enumerate(shot_boundaries):
            print(f"边界{idx + 1}：帧对{boundary['帧对']} | 时间{boundary['视频时间（秒）']}s | 差值{boundary['差值']}")

    return shot_boundaries


# ===================== 主函数 =====================
if __name__ == "__main__":
    extract_success = extract_frames()
    if extract_success:
        print_and_show_extract_result()

    hist_diff, frame_times, frame_files, frame_indices = calc_hist_diff()
    shot_boundaries = detect_shot_boundary(hist_diff, frame_times, frame_files, frame_indices)
    shots = split_shots_by_frames(hist_diff, frame_files, frame_indices, THRESHOLD)
    print_shot_result(shots)