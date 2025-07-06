import argparse
import json
import os
import struct
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import redis
from easydict import EasyDict

from cam_base.camera_redis_interface import CameraRedisPubInterface

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--camera-ref", type=str, required=True)
    parser.add_argument("--camera-address", default="/dev/video0", type=str)
    parser.add_argument("--use-rgb", action="store_true")
    parser.add_argument("--use-depth", action="store_true")
    
    parser.add_argument("--img-w", default=640, type=int)
    parser.add_argument("--img-h", default=480, type=int)
    parser.add_argument("--fps", default=30, type=int)
    
    parser.add_argument("--rgb-convention", default="rgb", choices=["bgr", "rgb"])
    parser.add_argument("--visualization", action="store_true")

    # parser.add_argument("--publish-freq", default=50, type=int, help="Redis publish frequency in Hz")
    parser.add_argument("--sync-with-robot", action="store_true", help="Sync with robot control frequency")
    args = parser.parse_args()

    # # Set publish frequency based on sync with robot or user input
    # if args.sync_with_robot:
    #     freq = 50.0  # Default frequency for sync with robot
    # else:
    #     freq = float(args.publish_freq)
    
    # print(f"Camera publish frequency set to: {freq} Hz")

    # # ...existing camera setup code...

    # print("Starting camera publisher...")
    
    img_counter = 0
    
    # Constants
    MAX_IMG_NUM = 653360
    COUNT_THRESH = 5
    counter = COUNT_THRESH


    # Parse camera reference
    parts = args.camera_ref.split('_')
    camera_type = parts[0]
    camera_id = parts[1] if len(parts) > 1 else "0"
    camera_name = args.camera_ref

    # Create camera info
    camera_info = EasyDict({
        "camera_id": camera_id,
        "camera_name": camera_name,
        "camera_type": camera_type
    })

    print(f"Starting camera node for {camera_name} (ID: {camera_id}, Type: {camera_type})")
    
    # Initialize Redis interface
    camera2redis_pub_interface = CameraRedisPubInterface(
        camera_info=camera_info,
        redis_host=args.host, 
        redis_port=args.port
    )

    # Node configuration
    node_config = EasyDict(use_color=args.use_rgb, use_depth=args.use_depth)
    
    # Initialize camera
    pipeline = None
    cap = None
    
    if camera_type == "rs":
        # RealSense camera
        try:
            import pyrealsense2 as rs
            
            pipeline = rs.pipeline()
            config = rs.config()
            
            if node_config.use_color:
                config.enable_stream(rs.stream.color, args.img_w, args.img_h, rs.format.bgr8, args.fps)
            if node_config.use_depth:
                config.enable_stream(rs.stream.depth, args.img_w, args.img_h, rs.format.z16, args.fps)
            
            profile = pipeline.start(config)

            # Set camera options for better performance
            color_sensor = profile.get_device().first_color_sensor()
            if color_sensor.supports(rs.option.auto_exposure_priority):
                color_sensor.set_option(rs.option.auto_exposure_priority, 0)  
            if color_sensor.supports(rs.option.exposure):
                color_sensor.set_option(rs.option.exposure, 100) 
            
            print("RealSense camera initialized with optimized settings")

#             pipeline.start(config)
            print("RealSense camera initialized successfully")
            
            def get_last_obs():
                try:
                    frames = pipeline.wait_for_frames()
                    result = {}
                    
                    if node_config.use_color:
                        color_frame = frames.get_color_frame()
                        if color_frame:
                            result["color"] = np.asanyarray(color_frame.get_data())
                    
                    if node_config.use_depth:
                        depth_frame = frames.get_depth_frame()
                        if depth_frame:
                            result["depth"] = np.asanyarray(depth_frame.get_data())
                    
                    return result if result else None
                except Exception as e:
                    print(f"Error getting frame: {e}")
                    return None
                
        except ImportError:
            print("pyrealsense2 not installed")
            return


    elif camera_type in ["webcam", "gopro"]:
        # Always use the camera address for webcam/gopro
        device_id = args.camera_address
        
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            print(f"Error: Cannot open camera at {device_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.img_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.img_h)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        print(f"OpenCV camera initialized successfully for {camera_type}")
        
        def get_last_obs():
            ret, frame = cap.read()
            if ret:
                return {"color": frame}
            return None

    # Create save directory
    t = time.time()
    save_dir = f"demos_collected/images/{camera_type}_{camera_name}_{t}"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Starting camera publisher...")
    
    img_counter = 0
    # freq = 10.0  # publish frequency in Hz
    MAX_IMG_NUM = 653360
    COUNT_THRESH = 5
    counter = COUNT_THRESH

    print("Camera publisher started... Press Ctrl+C to stop")
    
    try:
        while True:
            start_time = time.time_ns()

            # Get frame from camera
            capture = get_last_obs()
            if capture is None:
                continue

            # Check if we need to save the image
            save_img = camera2redis_pub_interface.get_save_img_info()
            
            if save_img:
                counter = 0
            else:
                counter += 1

            # Prepare image info
            t = time.time_ns()
            img_info = {
                "color_img_name": "",
                "depth_img_name": "",
                "time": t,
                "camera_type": camera_type,
                "intrinsics": {"color": [], "depth": []}
            }

            # Prepare images
            imgs = {}
            
            if node_config.use_color and "color" in capture and capture["color"] is not None:
                color_img = capture["color"]
                
                # Color image processing
                if args.rgb_convention == "rgb":
                    imgs["color"] = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                else:
                    imgs["color"] = color_img
                
                color_img_name = f"{save_dir}/color_{img_counter:09d}"
                img_info["color_img_name"] = color_img_name
                    
            if node_config.use_depth and "depth" in capture and capture["depth"] is not None:
                imgs["depth"] = capture["depth"]
                depth_img_name = f"{save_dir}/depth_{img_counter:09d}"
                img_info["depth_img_name"] = depth_img_name

            # Publish to Redis
            camera2redis_pub_interface.set_img_info(img_info)
            camera2redis_pub_interface.set_img_buffer(imgs=imgs)

            # Update counter
            img_counter += 1
            img_counter = img_counter % MAX_IMG_NUM

            # Visualization
            if args.visualization:
                if "color" in imgs:
                    display_img = imgs["color"]
                    if args.rgb_convention == "rgb":
                        display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)

                    scale = 0.25  # half size
                    h, w = display_img.shape[:2]
                    new_size = (int(w*scale), int(h*scale))
                    display_img = cv2.resize(display_img, new_size)
                    cv2.imshow(f"Camera {camera_name}", display_img)
                    
                if "depth" in imgs:
                    depth_display = (imgs["depth"] * 0.001).astype(np.float32)
                    cv2.imshow(f"Depth {camera_name}", depth_display)
                    
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # Control publish frequency
            # end_time = time.time_ns()
            # time_interval = (end_time - start_time) / (10 ** 9)
            # if time_interval < 1.0 / freq:
            #     time.sleep(1.0 / freq - time_interval)
            
            # print(f"The camera node took {time_interval} to transmit image")

            # Check if we need to finish
            if camera2redis_pub_interface.finished:
                break

    except KeyboardInterrupt:
        print("\nStopping camera publisher...")
    
    finally:
        if pipeline:
            pipeline.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("Camera publisher stopped")

if __name__ == "__main__":
    main()