import os
import imageio.v2 as imageio
from pathlib import Path
from tqdm import tqdm


def images_to_video(image_dir, out_path, fps=15, quality=10):
    """将一个文件夹中的 PNG 序列合成为 MP4 视频。

    Args:
        image_dir: 包含 PNG 图片的文件夹路径
        out_path:  输出 MP4 文件路径
        fps:       帧率
        quality:   视频质量 (0-10, 越高越好)
    """
    # 按文件名排序，确保 00000 -> 00148 的正确顺序
    files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])

    if len(files) == 0:
        print(f"  [跳过] {image_dir} 中没有找到 PNG 文件")
        return

    print(f"  共 {len(files)} 帧，正在读取...")
    frames = []
    for f in tqdm(files, desc=f"读取 {Path(image_dir).name}"):
        img = imageio.imread(os.path.join(image_dir, f))
        frames.append(img)

    # 合成为 MP4 (H.264 编码)
    imageio.mimwrite(out_path, frames, fps=fps, quality=quality,
                     codec='libx264', macro_block_size=1)
    print(f"  已保存: {out_path}")


if __name__ == "__main__":
    root = Path(__file__).parent  # assets 目录

    tasks = [
        (root / "scene5" / "gt",      root / "scene5_gt.mp4"),
        (root / "scene5" / "renders", root / "scene5_renders.mp4"),
        (root / "scene7" / "gt",      root / "scene7_gt.mp4"),
        (root / "scene7" / "renders", root / "scene7_renders.mp4"),
    ]

    print("=" * 50)
    print("开始合成视频")
    print("=" * 50)

    for img_dir, out_path in tasks:
        print(f"\n处理: {img_dir.name} -> {out_path.name}")
        images_to_video(str(img_dir), str(out_path), fps=15, quality=10)

    print("\n" + "=" * 50)
    print("全部完成！输出文件：")
    for _, out_path in tasks:
        print(f"  {out_path}")
    print("=" * 50)
