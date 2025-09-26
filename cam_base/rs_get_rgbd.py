#!/usr/bin/env python3
"""
Capture a synchronized RGB + Depth frame from an Intel RealSense camera,
save RGB to JPG and Depth to 16-bit PNG, then export a colored point cloud to PLY.

Requires:
  - pyrealsense2
  - numpy
  - opencv-python

Usage (example):
  python realsense_capture_to_ply.py --color-out color.jpg --depth-out depth.png --ply-out cloud.ply \
                                     --width 1280 --height 720 --fps 30 --warmup 30
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import cv2
import pyrealsense2 as rs


def save_intrinsics(intrin, path: Path):
    data = {
        "width": intrin.width,
        "height": intrin.height,
        "ppx": intrin.ppx,
        "ppy": intrin.ppy,
        "fx": intrin.fx,
        "fy": intrin.fy,
        "model": str(intrin.model),
        "coeffs": list(intrin.coeffs),
    }
    path.write_text(json.dumps(data, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color-out", default="color.jpg", help="output color image (JPG)")
    ap.add_argument("--depth-out", default="depth.png", help="output depth image (16-bit PNG)")
    ap.add_argument("--ply-out",   default="cloud.ply", help="output colored point cloud (PLY)")
    ap.add_argument("--width",  type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps",    type=int, default=30)
    ap.add_argument("--warmup", type=int, default=30, help="frames to skip for auto-exposure to settle")
    ap.add_argument("--device-serial", default=None, help="optional device serial if multiple cameras")
    args = ap.parse_args()

    color_path = Path(args.color_out)
    depth_path = Path(args.depth_out)
    ply_path   = Path(args.ply_out)
    intrin_path = ply_path.with_suffix(".intrinsics.json")

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    if args.device_serial:
        config.enable_device(args.device_serial)

    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    profile = pipeline.start(config)

    # Get depth scale (for info)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[Info] Depth scale: {depth_scale:.6f} meters per unit")

    # Align depth to color
    align = rs.align(rs.stream.color)

    # Warm up for stable exposure/white balance
    print(f"[Info] Warming up for {args.warmup} frames...")
    for _ in range(max(0, args.warmup)):
        pipeline.wait_for_frames()

    print("[Info] Capturing one aligned frameset...")
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()

    if not depth_frame or not color_frame:
        pipeline.stop()
        raise RuntimeError("Failed to acquire both depth and color frames.")

    # Save intrinsics (helpful for later reprojection)
    color_stream = color_frame.profile.as_video_stream_profile()
    color_intrinsics = color_stream.get_intrinsics()
    save_intrinsics(color_intrinsics, intrin_path)
    print(f"[OK] Saved color intrinsics to {intrin_path}")

    # Convert to numpy
    depth_image = np.asanyarray(depth_frame.get_data())  # dtype=uint16, in depth units
    color_image = np.asanyarray(color_frame.get_data())  # dtype=uint8, BGR

    # Save images (depth stays 16-bit PNG)
    # Note: cv2.imwrite preserves dtype; uint16 -> 16-bit PNG.
    cv2.imwrite(str(color_path), color_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(depth_path), depth_image)
    print(f"[OK] Saved color JPG to {color_path}")
    print(f"[OK] Saved depth PNG (16-bit) to {depth_path}")

    # Build colored point cloud using official RealSense API
    pc = rs.pointcloud()
    pc.map_to(color_frame)                  # attach color to points
    points = pc.calculate(depth_frame)   # compute point cloud from depth

    # Export to PLY with color
    points.export_to_ply(str(ply_path), color_frame)
    print(f"[OK] Exported colored point cloud to {ply_path}")

    pipeline.stop()
    print("[Done]")


if __name__ == "__main__":
    main()
