#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
zed_get_rgb_K.py
Capture rectified, undistorted left/right RGB images and export intrinsics & baseline.

Requirements:
  - ZED SDK (v4/v5) with Python API: pyzed.sl
  - OpenCV (cv2)
  - imageio (or Pillow) for writing PNGs
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import cv2
import imageio.v3 as iio

import pyzed.sl as sl


def get_baseline_m(calib) -> float:
    """
    Try to robustly read the stereo baseline (meters) across ZED SDK versions.
    Baseline is the X translation from left to right camera.
    """
    # SDK 5.x exposes stereo_transform (sl.Transform) with translation
    if hasattr(calib, "stereo_transform") and calib.stereo_transform is not None:
        try:
            # sl.Transform -> get_translation() -> sl.Translation -> get() -> [x, y, z]
            tr = calib.stereo_transform.get_translation()
            v = tr.get() if hasattr(tr, "get") else [getattr(tr, "x", 0.0), 0.0, 0.0]
            return float(abs(v[0]))
        except Exception:
            pass

    # Older SDKs expose T as a vector/array/translation
    if hasattr(calib, "T") and calib.T is not None:
        T = calib.T
        # Try common access patterns
        for getter in (
            lambda: float(abs(T[0])),
            lambda: float(abs(T.get()[0])),
            lambda: float(abs(getattr(T, "x", 0.0))),
            lambda: float(abs(getattr(T, "tx", 0.0))),
        ):
            try:
                return getter()
            except Exception:
                continue

    raise RuntimeError("Could not read stereo baseline (translation X) from calibration parameters.")


def bgra_to_rgb(img_bgra: np.ndarray) -> np.ndarray:
    """Convert BGRA uint8 image to RGB uint8 (3 channels)."""
    if img_bgra.ndim == 3 and img_bgra.shape[2] == 4:
        rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGB)
        return np.ascontiguousarray(rgb)
    elif img_bgra.ndim == 3 and img_bgra.shape[2] == 3:
        # Already 3 channels; assume BGR from OpenCV, convert to RGB
        rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb)
    else:
        raise ValueError(f"Unexpected image shape {img_bgra.shape} (expected HxWx4 BGRA or HxWx3 BGR).")


def main():
    out_left = Path("left.png")
    out_right = Path("right.png")
    out_json = Path("calib.json")

    # --- Open camera
    init = sl.InitParameters()
    # We don't need depth; ensure rectified RGB retrieval
    init.depth_mode = sl.DEPTH_MODE.NONE
    # Units for baseline (meters)
    init.coordinate_units = sl.UNIT.METER
    # Use default color stream resolution/fps (change if you want)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.camera_fps = 15

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] ZED open failed: {status}. Is the camera connected and SDK installed?", file=sys.stderr)
        sys.exit(1)

    try:
        # Let auto-exposure stabilize a bit (optional but helpful)
        runtime = sl.RuntimeParameters()
        for _ in range(8):
            if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                time.sleep(0.01)

        # Final grab for capture
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to grab a frame from ZED.")

        # --- Retrieve rectified left/right images (BGRA by default)
        left_mat, right_mat = sl.Mat(), sl.Mat()
        cam.retrieve_image(left_mat, sl.VIEW.LEFT)    # rectified, undistorted
        cam.retrieve_image(right_mat, sl.VIEW.RIGHT)  # rectified, undistorted
        # cam.retrieve_image(left_mat, sl.VIEW.LEFT_UNRECTIFIED)    # unrectified, undistorted
        # cam.retrieve_image(right_mat, sl.VIEW.RIGHT_UNRECTIFIED)  # unrectified, undistorted


        left_bgra = left_mat.get_data()
        right_bgra = right_mat.get_data()

        # Convert to RGB (3 channels)
        left_rgb = bgra_to_rgb(left_bgra)
        right_rgb = bgra_to_rgb(right_bgra)
        # import pdb; pdb.set_trace()
        # Save as PNG (3 channels)
        iio.imwrite(out_left.as_posix(), left_rgb)
        iio.imwrite(out_right.as_posix(), right_rgb)

        # --- Intrinsics & baseline from rectified LEFT camera
        info = cam.get_camera_information()
        # import ipdb; ipdb.set_trace()
        calib = info.camera_configuration.calibration_parameters # SDK5.0
        # calib = info.calibration_parameters # SDK4.0
        left_cam = calib.left_cam  # rectified left intrinsics

        fx = float(left_cam.fx)
        fy = float(left_cam.fy)
        cx = float(left_cam.cx)
        cy = float(left_cam.cy)

        H, W = left_rgb.shape[:2]
        K = [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]]

        baseline_m = get_baseline_m(calib)

        # Print to console
        print("Saved rectified RGB images -> left.png, right.png")
        print(f"Image size: {W} x {H}")
        print("Left camera intrinsics (K):")
        print(np.array(K))
        print(f"Stereo baseline: {baseline_m:.6f} m ({baseline_m*1000:.3f} mm)")

        # Save to JSON
        meta = {
            "image_size": {"width": W, "height": H},
            "left_intrinsics": {
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "K": K
            },
            "baseline_m": baseline_m,
            "notes": "Images are rectified & undistorted (epipolar lines horizontal). Saved as 3-channel RGB PNG."
        }
        out_json.write_text(json.dumps(meta, indent=2))
        print(f"Wrote intrinsics/baseline -> {out_json}")

    finally:
        cam.close()


if __name__ == "__main__":
    main()
