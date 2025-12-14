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
FPS = 5  # 与解析视频时的帧率一致
THRESHOLD = 0.8  # 直方图差值阈值（可根据实际数据调整）


def extract_frames():
    os.makedirs(FRAME_DIR, exist_ok=True)
    # 构造FFmpeg命令（补全可执行文件路径）
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
    # 读取并转换为RGB格式（避免Matplotlib显示异常）
    img = Image.open(frame_path).convert('RGB')
    img_array = np.array(img)
    pixel_mean = np.mean(img_array)  # 计算像素均值（仅展示，不做黑帧判断）

    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制帧画面
    plt.figure(figsize=(8, 6))
    plt.imshow(img_array)
    plt.title(f"{title}（像素均值：{pixel_mean:.2f}）", fontsize=12)
    plt.axis("off")  # 隐藏坐标轴
    plt.tight_layout()  # 优化布局
    plt.show()

    # 打印帧信息
    print(f"【{title}】")
    print(f"  文件名称：{frame_name}")
    print(f"  文件路径：{frame_path}")
    print(f"  像素均值：{pixel_mean:.2f}\n")


def print_and_show_extract_result():
    """提取完成后打印统计信息，展示首/中/尾帧"""
    # 获取所有帧文件并按编号排序
    frame_files = sorted([
        f for f in os.listdir(FRAME_DIR)
        if f.startswith("frame_") and f.endswith(".jpg")
    ])

    # 校验帧文件数量
    if not frame_files:
        print("=== 提取结果异常 ===")
        print("未检测到任何提取的帧文件！")
        return

    # 计算基础统计信息
    total_frames = len(frame_files)
    video_duration = total_frames / FPS  # 估算视频时长

    # 打印提取结果统计
    print("=== 帧提取结果统计 ===")
    print(f"总提取帧数：{total_frames} 帧")
    print(f"提取帧率：{FPS} fps")
    print(f"估算视频时长：{video_duration:.2f} 秒")
    print(f"帧文件范围：{frame_files[0]} ~ {frame_files[-1]}\n")

    # 选择首帧、中间帧、末尾帧
    first_frame = frame_files[0]
    middle_frame = frame_files[total_frames // 2]
    last_frame = frame_files[-1]

    # 展示关键帧
    print("=== 展示关键帧 ===")
    show_frame(first_frame, "首帧")
    show_frame(middle_frame, "中间帧")
    show_frame(last_frame, "末尾帧")

def calc_frame_hist(frame_path):
    frame = cv2.imread(frame_path)
    # 增加容错：处理图片读取失败的情况
    if frame is None:
        print(f"警告：无法读取帧文件 {frame_path}，返回空直方图")
        return np.zeros((18*8*8,))  # 返回与原直方图维度一致的空数组
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, [18, 8, 8],
        [0, 180, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def calc_hist_diff():
    # 获取排序后的帧文件
    frame_files = sorted(
        [f for f in os.listdir(FRAME_DIR) if f.startswith("frame_") and f.endswith(".jpg")])  # 增加后缀过滤，避免非jpg文件干扰
    if len(frame_files) < 2:
        print("帧数量不足，无法计算差值！")
        exit(1)

    hist_diff = []
    frame_times = []  # 帧对应的视频时间（秒）
    # 计算相邻帧差值（卡方距离）
    for i in range(1, len(frame_files)):
        hist_prev = calc_frame_hist(os.path.join(FRAME_DIR, frame_files[i - 1]))
        hist_curr = calc_frame_hist(os.path.join(FRAME_DIR, frame_files[i]))
        # 卡方距离：越大表示帧差异越大
        diff = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CHISQR)
        hist_diff.append(diff)
        # 修复：帧时间取「后一帧的时间」，且确保帧编号解析正确
        frame_name = frame_files[i]
        try:
            # 提取帧编号（如frame_0005.jpg → 5）
            frame_idx = int(frame_name.split("_")[1].split(".")[0])
            frame_time = frame_idx / FPS  # 第N帧对应的视频时间 = 帧编号 / 帧率
            frame_times.append(frame_time)
        except (IndexError, ValueError):
            print(f"警告：帧文件命名异常 {frame_name}，跳过时间计算")
            frame_times.append(i / FPS)  # 兜底：用循环索引估算时间

    return hist_diff, frame_times, frame_files


def detect_shot_boundary(hist_diff, frame_times, frame_files):
    # 1. 绘制直方图差值柱状图
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(12, 6))

    # 修复1：x轴长度与hist_diff严格匹配（相邻帧对数量 = 总帧数-1）
    x = np.arange(len(hist_diff))
    ax.bar(x, hist_diff, color='skyblue', label='帧间直方图差值')

    # 绘制阈值线
    ax.axhline(y=THRESHOLD, color='red', linestyle='--', label=f'阈值 = {THRESHOLD}')

    # 修复2：x轴标签改为「帧对（前帧→后帧）」，避免时间标注越界
    ax.set_xlabel('相邻帧对（如 1→2 表示第1帧与第2帧的差值）')
    ax.set_ylabel('直方图卡方差值')
    ax.set_title('视频帧间直方图差值分布')
    ax.legend()

    # 修复3：合理设置刻度标注（按步长标注，且不超过数组长度）
    step = max(1, len(hist_diff) // 10)  # 最多显示10个刻度，避免拥挤
    xticks_pos = x[::step]
    xticks_labels = [f"{i + 1}→{i + 2}\n({frame_times[i]:.1f}s)" for i in xticks_pos]  # 标注帧对+时间
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_labels, rotation=0)

    plt.tight_layout()
    plt.savefig("hist_diff_bar.png")
    plt.show()

    # 2. 检测镜头边界（差值超过阈值的位置）
    shot_boundaries = []
    for i, diff in enumerate(hist_diff):
        if diff > THRESHOLD:
            boundary_time = frame_times[i]
            boundary_frame_pair = f"{i + 1}→{i + 2}"  # 明确是哪一对帧的边界
            shot_boundaries.append({
                "帧对": boundary_frame_pair,
                "视频时间（秒）": round(boundary_time, 1),
                "差值": round(diff, 3)
            })

    # 输出检测结果（优化展示）
    print("\n检测到的镜头边界：")
    if not shot_boundaries:
        print("未检测到超过阈值的镜头边界！")
    else:
        for idx, boundary in enumerate(shot_boundaries):
            print(f"边界{idx + 1}：帧对{boundary['帧对']} | 时间{boundary['视频时间（秒）']}s | 差值{boundary['差值']}")

    return shot_boundaries


# ===================== 6. 主函数 =====================
if __name__ == "__main__":

    # 执行帧提取
    extract_success = extract_frames()
    # 打印并展示提取结果
    if extract_success:
        print_and_show_extract_result()

    hist_diff, frame_times, frame_files = calc_hist_diff()

    shot_boundaries = detect_shot_boundary(hist_diff, frame_times, frame_files)