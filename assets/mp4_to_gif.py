"""批量将 assets 目录下的 MP4 演示视频转换为 GIF。

使用 imageio-ffmpeg 自带的 ffmpeg 二进制，无需单独安装 ffmpeg。
采用两遍法（palettegen + paletteuse），画质优于直接转换。

用法:
    python assets/mp4_to_gif.py                            # 默认 fps=10, width=480
    python assets/mp4_to_gif.py --fps 15 --width 640       # 更高清、更大体积
    python assets/mp4_to_gif.py --fps 8  --width 360       # 更小体积
"""

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg


def mp4_to_gif(ffmpeg_exe, in_path, out_path, fps=10, width=480):
    """将单个 MP4 转换为 GIF（两遍法）。

    Args:
        ffmpeg_exe: ffmpeg 可执行文件路径
        in_path:    输入 MP4 路径
        out_path:   输出 GIF 路径
        fps:        GIF 帧率（越低体积越小）
        width:      GIF 宽度像素（高度按比例缩放）
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    palette = out_path.with_suffix(".palette.png")

    # 第一遍：生成最优调色板
    vf_gen = f"fps={fps},scale={width}:-1:flags=lanczos,palettegen"
    cmd_gen = [ffmpeg_exe, "-i", str(in_path), "-vf", vf_gen, "-y", str(palette)]

    # 第二遍：用调色板量化生成 GIF
    vf_use = f"fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse"
    cmd_use = [ffmpeg_exe, "-i", str(in_path), "-i", str(palette),
               "-filter_complex", vf_use, "-y", str(out_path)]

    try:
        subprocess.run(cmd_gen, check=True, capture_output=True)
        subprocess.run(cmd_use, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"  [错误] ffmpeg 失败:\n{e.stderr.decode(errors='ignore')}")
        raise
    finally:
        if palette.exists():
            palette.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量将 assets 下的 MP4 转为 GIF")
    parser.add_argument("--fps", type=int, default=10, help="GIF 帧率 (默认 10)")
    parser.add_argument("--width", type=int, default=480, help="GIF 宽度像素 (默认 480)")
    args = parser.parse_args()

    root = Path(__file__).parent  # assets 目录
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    tasks = [
        (root / "eating_gt.mp4",      root / "eating_gt.gif"),
        (root / "eating_renders.mp4", root / "eating_renders.gif"),
        (root / "cycling_gt.mp4",      root / "cycling_gt.gif"),
        (root / "cycling_renders.mp4", root / "cycling_renders.gif"),
    ]

    print("=" * 50)
    print(f"开始转换 MP4 -> GIF  (fps={args.fps}, width={args.width})")
    print(f"ffmpeg: {ffmpeg_exe}")
    print("=" * 50)

    for in_path, out_path in tasks:
        if not in_path.exists():
            print(f"\n[跳过] 输入文件不存在: {in_path}")
            continue
        print(f"\n转换: {in_path.name} -> {out_path.name}")
        mp4_to_gif(ffmpeg_exe, in_path, out_path, fps=args.fps, width=args.width)
        size_kb = out_path.stat().st_size / 1024
        print(f"  已保存: {out_path.name}  ({size_kb:.0f} KB)")

    print("\n" + "=" * 50)
    print("全部完成！输出文件：")
    for _, out_path in tasks:
        if out_path.exists():
            size_kb = out_path.stat().st_size / 1024
            print(f"  {out_path.name}  ({size_kb:.0f} KB)")
    print("=" * 50)
