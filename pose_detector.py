import cv2
import numpy as np
import json
from ultralytics import YOLO
import os
import time
from pathlib import Path
from performance_metrics import metrics_collector

class PoseDetector:
    def __init__(self, model_path='yolo11n-pose.pt'):
        """
        Khởi tạo YOLOv11 pose detector
        
        Args:
            model_path: Đường dẫn đến model YOLOv11 (auto-download nếu không tồn tại)
        """
        print(f"🤖 Initializing YOLO Pose Detector...")
        
        try:
            self.model = YOLO(model_path)
            print(f"✅ Loaded YOLO model: {model_path}")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            raise
        
        # Confidence thresholds có thể điều chỉnh
        self.person_conf_threshold = 0.2    # Person detection confidence
        self.keypoint_conf_threshold = 0.15 # Keypoint visibility confidence
        
        # Mapping YOLO keypoints to COCO format (17 keypoints)
        # YOLO sử dụng format khác một chút so với COCO standard
        self.yolo_to_coco = {
            0: 0,   # nose -> nose
            1: 2,   # left_eye -> right_eye (YOLO perspective)
            2: 1,   # right_eye -> left_eye
            3: 4,   # left_ear -> right_ear
            4: 3,   # right_ear -> left_ear
            5: 6,   # left_shoulder -> right_shoulder
            6: 5,   # right_shoulder -> left_shoulder
            7: 8,   # left_elbow -> right_elbow
            8: 7,   # right_elbow -> left_elbow
            9: 10,  # left_wrist -> right_wrist
            10: 9,  # right_wrist -> left_wrist
            11: 12, # left_hip -> right_hip
            12: 11, # right_hip -> left_hip
            13: 14, # left_knee -> right_knee
            14: 13, # right_knee -> left_knee
            15: 16, # left_ankle -> right_ankle
            16: 15  # right_ankle -> left_ankle
        }
        
        # COCO keypoint names for reference
        self.coco_keypoint_names = [
            'nose',           # 0
            'left_eye',       # 1
            'right_eye',      # 2
            'left_ear',       # 3
            'right_ear',      # 4
            'left_shoulder',  # 5
            'right_shoulder', # 6
            'left_elbow',     # 7
            'right_elbow',    # 8
            'left_wrist',     # 9
            'right_wrist',    # 10
            'left_hip',       # 11
            'right_hip',      # 12
            'left_knee',      # 13
            'right_knee',     # 14
            'left_ankle',     # 15
            'right_ankle'     # 16
        ]
        
        # Màu sắc cho visualization nhiều người
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
            (255, 192, 203), # Pink
            (165, 42, 42),  # Brown
            (0, 128, 0),    # Dark Green
            (128, 0, 0),    # Dark Red
            (0, 0, 128),    # Dark Blue
        ]
        
        # Skeleton connections cho visualization (COCO format)
        self.skeleton_connections = [
            # Head connections
            (0, 1), (0, 2), (1, 3), (2, 4),  # nose-eyes, eyes-ears
            
            # Upper body
            (5, 6),   # shoulders
            (5, 7), (7, 9),   # left arm
            (6, 8), (8, 10),  # right arm
            (5, 11), (6, 12), # shoulder to hip
            (11, 12), # hips
            
            # Lower body  
            (11, 13), (13, 15), # left leg
            (12, 14), (14, 16), # right leg
        ]
        
        print(f"🎯 Detection settings:")
        print(f"   Person confidence: {self.person_conf_threshold}")
        print(f"   Keypoint confidence: {self.keypoint_conf_threshold}")
    
    def set_confidence_thresholds(self, person_conf=None, keypoint_conf=None):
        """
        Điều chỉnh confidence thresholds
        
        Args:
            person_conf: Confidence threshold cho person detection (0.0-1.0)
            keypoint_conf: Confidence threshold cho keypoint visibility (0.0-1.0)
        """
        if person_conf is not None:
            self.person_conf_threshold = person_conf
            print(f"🎯 Person confidence threshold updated: {self.person_conf_threshold}")
        
        if keypoint_conf is not None:
            self.keypoint_conf_threshold = keypoint_conf
            print(f"🎯 Keypoint confidence threshold updated: {self.keypoint_conf_threshold}")
    
    def detect_poses_from_video(self, video_path, output_path=None, max_people=10, 
                               save_frames=False, frame_step=1):
        """
        Phát hiện poses từ video cho nhiều người
        
        Args:
            video_path: Đường dẫn video input
            output_path: Đường dẫn video output với pose overlay (optional)
            max_people: Số người tối đa để xử lý mỗi frame
            save_frames: Có save individual frames không
            frame_step: Bước nhảy giữa frames (1 = process all frames)
            
        Returns:
            tuple: (all_poses, width, height, fps, json_path)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"📹 Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Lấy thông tin video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"📊 Video info:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Total frames: {total_frames}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Frame step: {frame_step}")
        
        # Khởi tạo danh sách poses
        all_poses = []
        frame_idx = 0
        processed_frames = 0
        
        # Setup video writer nếu cần save output
        output_video_path = os.path.splitext(output_path)[0] + "_poses.mp4"
        video_writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps/frame_step, (width, height))
            print(f"💾 Will save annotated video to: {output_video_path}")
        
        # Setup frame saving
        frames_dir = None
        if save_frames:
            video_name = Path(video_path).stem
            frames_dir = f"{video_name}_frames"
            os.makedirs(frames_dir, exist_ok=True)
            print(f"📁 Will save frames to: {frames_dir}")
        
        print(f"\n🚀 Starting pose detection...")
        print(f"⚙️  Settings: person_conf={self.person_conf_threshold}, "
              f"keypoint_conf={self.keypoint_conf_threshold}, max_people={max_people}")
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames nếu frame_step > 1
            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue
            
            # Record start time for this frame
            frame_start_time = time.time()
            
            # Detect poses using YOLOv11
            results = self.model(
                frame, 
                verbose=False,
                conf=self.person_conf_threshold,  # Person detection confidence
                iou=0.5,                          # NMS threshold
                max_det=20                        # Max detections
            )
            
            # --- FILTERING LOGIC: Keep only the largest person if multiple are detected ---
            if results and len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.data) > 1:
                boxes_tensor = results[0].boxes.data
                
                best_person_idx = -1
                max_area = -1
                
                for i in range(len(boxes_tensor)):
                    box = boxes_tensor[i]
                    # box format: [x1, y1, x2, y2, conf, class]
                    x1, y1, x2, y2 = self._extract_value(box[0]), self._extract_value(box[1]), self._extract_value(box[2]), self._extract_value(box[3])
                    area = (x2 - x1) * (y2 - y1)
                    
                    if area > max_area:
                        max_area = area
                        best_person_idx = i
                
                # Filter the results object to only keep the best person
                if best_person_idx != -1:
                    results[0].keypoints.data = results[0].keypoints.data[best_person_idx:best_person_idx+1]
                    results[0].boxes.data = results[0].boxes.data[best_person_idx:best_person_idx+1]
            # --- END FILTERING LOGIC ---

            # Process detections
            frame_poses = self._process_detections(results, max_people)
            all_poses.append(frame_poses)
            
            # Visualize và save nếu cần
            if output_video_path or save_frames:
                annotated_frame = self._draw_poses_on_frame(frame.copy(), results, max_people)
                
                if video_writer:
                    video_writer.write(annotated_frame)
                
                if save_frames and frames_dir:
                    frame_filename = f"{frames_dir}/frame_{processed_frames:06d}.jpg"
                    cv2.imwrite(frame_filename, annotated_frame)
            
            processed_frames += 1
            frame_idx += 1

            # Record frame processing metrics
            processing_time = time.time() - frame_start_time
            people_count = len(frame_poses)
            metrics_collector.record_frame_processing(processed_frames, processing_time, people_count)
            
            # Progress update
            if processed_frames % 50 == 0:
                detected_people = len(frame_poses)
                elapsed = time.time() - start_time
                fps_current = processed_frames / elapsed
                eta = (total_frames/frame_step - processed_frames) / fps_current if fps_current > 0 else 0
                
                print(f"📈 Progress: {processed_frames}/{total_frames//frame_step} frames "
                      f"({processed_frames/(total_frames//frame_step)*100:.1f}%) | "
                      f"Speed: {fps_current:.1f} FPS | "
                      f"People: {detected_people} | "
                      f"ETA: {eta:.1f}s")
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        
        # Calculate final stats
        total_time = time.time() - start_time
        avg_fps = processed_frames / total_time
        
        print(f"\n✅ Detection completed!")
        print(f"⏱️  Processing time: {total_time:.2f}s")
        print(f"🚀 Average speed: {avg_fps:.2f} FPS")
        print(f"📊 Processed {processed_frames} frames")
        
        # Save poses to JSON
        json_path = self._save_poses_to_json(
            all_poses, video_path, output_path, width, height, fps/frame_step, max_people, frame_step
        )
        
        # Print detection statistics
        self._print_detection_statistics(all_poses, processed_frames)
        
        return all_poses, width, height, fps/frame_step, json_path
    
    def detect_poses_from_image(self, image_path, output_path=None, max_people=10):
        """
        Phát hiện poses từ single image
        
        Args:
            image_path: Đường dẫn image
            output_path: Đường dẫn output image với poses
            max_people: Số người tối đa
            
        Returns:
            list: Poses detected trong image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # Detect poses
        results = self.model(
            image,
            verbose=False,
            conf=self.person_conf_threshold,
            iou=0.5,
            max_det=20
        )
        
        # Process detections
        poses = self._process_detections(results, max_people)
        
        # Save annotated image if requested
        if output_path:
            annotated_image = self._draw_poses_on_frame(image.copy(), results, max_people)
            cv2.imwrite(output_path, annotated_image)
            print(f"💾 Saved annotated image to: {output_path}")
        
        print(f"📸 Detected {len(poses)} people in image")
        
        return poses
    
    def _process_detections(self, results, max_people):
        """Xử lý detection results để extract poses"""
        frame_poses = []
        
        if results and len(results) > 0:
            keypoints = results[0].keypoints
            
            if keypoints is not None and len(keypoints.data) > 0:
                for person_idx, person_keypoints in enumerate(keypoints.data):
                    if person_idx >= max_people:
                        break
                    
                    # Convert YOLO format to COCO format
                    person_pose = np.zeros((17, 3))  # 17 COCO keypoints, (x, y, confidence)
                    
                    for yolo_idx, coco_idx in self.yolo_to_coco.items():
                        if yolo_idx < len(person_keypoints):
                            kpt = person_keypoints[yolo_idx]
                            x = self._extract_value(kpt[0])
                            y = self._extract_value(kpt[1])
                            conf = self._extract_value(kpt[2])
                            
                            person_pose[coco_idx] = [x, y, conf]
                    
                    frame_poses.append(person_pose.tolist())
        
        return frame_poses
    
    def _extract_value(self, tensor_value):
        """Extract float value from tensor hoặc number"""
        if hasattr(tensor_value, 'cpu'):
            return float(tensor_value.cpu().numpy())
        elif hasattr(tensor_value, 'item'):
            return float(tensor_value.item())
        else:
            return float(tensor_value)
    
    def _draw_poses_on_frame(self, frame, results, max_people):
        """Vẽ poses lên frame để visualization"""
        if not results or len(results) == 0:
            return frame
        
        keypoints = results[0].keypoints
        if keypoints is None or len(keypoints.data) == 0:
            return frame
        
        for person_idx, person_keypoints in enumerate(keypoints.data):
            if person_idx >= max_people:
                break
            
            color = self.colors[person_idx % len(self.colors)]
            
            # Draw keypoints
            valid_keypoints = []
            for i, kpt in enumerate(person_keypoints):
                conf = self._extract_value(kpt[2])
                if conf > self.keypoint_conf_threshold:
                    x = int(self._extract_value(kpt[0]))
                    y = int(self._extract_value(kpt[1]))
                    
                    # Draw keypoint
                    cv2.circle(frame, (x, y), 4, color, -1)
                    cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)  # White border
                    
                    # Store for skeleton drawing
                    valid_keypoints.append((i, x, y, conf))
            
            # Draw skeleton connections
            self._draw_skeleton(frame, person_keypoints, color)
            
            # Draw person ID và confidence
            if len(person_keypoints) > 0:
                # Try to use nose position for label, fallback to first valid keypoint
                label_pos = None
                nose_kpt = person_keypoints[0]  # nose keypoint
                nose_conf = self._extract_value(nose_kpt[2])
                
                if nose_conf > self.keypoint_conf_threshold:
                    x = int(self._extract_value(nose_kpt[0]))
                    y = int(self._extract_value(nose_kpt[1]))
                    label_pos = (x, y)
                elif valid_keypoints:
                    # Use first valid keypoint
                    _, x, y, _ = valid_keypoints[0]
                    label_pos = (x, y)
                
                if label_pos:
                    # Draw background rectangle for text
                    text = f"Person {person_idx}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, 
                                (label_pos[0]-5, label_pos[1]-25),
                                (label_pos[0]+text_size[0]+5, label_pos[1]-5),
                                color, -1)
                    
                    # Draw text
                    cv2.putText(frame, text, (label_pos[0], label_pos[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _draw_skeleton(self, frame, person_keypoints, color):
        """Vẽ skeleton connections"""
        # Convert keypoints to format for easy access
        keypoints_dict = {}
        for i, kpt in enumerate(person_keypoints):
            conf = self._extract_value(kpt[2])
            if conf > self.keypoint_conf_threshold:
                x = int(self._extract_value(kpt[0]))
                y = int(self._extract_value(kpt[1]))
                
                # Map YOLO index to COCO index
                if i in self.yolo_to_coco:
                    coco_idx = self.yolo_to_coco[i]
                    keypoints_dict[coco_idx] = (x, y)
        
        # Draw skeleton connections (using COCO indices)
        for start_idx, end_idx in self.skeleton_connections:
            if start_idx in keypoints_dict and end_idx in keypoints_dict:
                start_pos = keypoints_dict[start_idx]
                end_pos = keypoints_dict[end_idx]
                
                cv2.line(frame, start_pos, end_pos, color, 2)
    
    def _save_poses_to_json(self, all_poses, video_path, output_path, width, height, fps, max_people, frame_step):
        """Save poses data to JSON file"""
        pose_data = {
            'poses': all_poses,
            'video_info': {
                'source_video': os.path.basename(video_path),
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': len(all_poses),
                'frame_step': frame_step
            },
            'detection_settings': {
                'person_conf_threshold': self.person_conf_threshold,
                'keypoint_conf_threshold': self.keypoint_conf_threshold,
                'max_people_per_frame': max_people,
                'model': 'yolo11n-pose.pt'
            },
            'data_format': {
                'keypoint_format': 'COCO_17',
                'coordinate_format': 'pixel_coordinates',
                'data_structure': 'frames -> people -> keypoints -> [x, y, confidence]',
                'keypoint_names': self.coco_keypoint_names
            },
            'metadata': {
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'version': '1.0',
                'compatible_with': 'VideoPose3D'
            }
        }
        
        # Tạo tên file JSON
        json_path = os.path.splitext(output_path)[0] + "_poses.json"
        
        try:
            with open(json_path, 'w') as f:
                json.dump(pose_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            print(f"💾 Saved 2D poses to: {json_path}")
        except Exception as e:
            print(f"❌ Error saving JSON: {e}")
            raise
        
        return json_path
    
    def _print_detection_statistics(self, all_poses, total_frames):
        """In thống kê về detection results"""
        if not all_poses:
            print("⚠️  No poses detected!")
            return
        
        total_people = sum(len(frame_poses) for frame_poses in all_poses)
        avg_people_per_frame = total_people / len(all_poses) if all_poses else 0
        frames_with_detections = sum(1 for frame_poses in all_poses if len(frame_poses) > 0)
        detection_rate = (frames_with_detections / len(all_poses)) * 100 if all_poses else 0
        
        print(f"\n📊 Detection Statistics:")
        print(f"   Total frames: {total_frames}")
        print(f"   Frames with detections: {frames_with_detections}")
        print(f"   Detection rate: {detection_rate:.1f}%")
        print(f"   Total person detections: {total_people}")
        print(f"   Average people per frame: {avg_people_per_frame:.2f}")
        
        if all_poses:
            max_people_in_frame = max(len(frame_poses) for frame_poses in all_poses)
            print(f"   Maximum people in single frame: {max_people_in_frame}")
            
            # Phân bố số người per frame
            people_distribution = {}
            for frame_poses in all_poses:
                count = len(frame_poses)
                people_distribution[count] = people_distribution.get(count, 0) + 1
            
            print(f"\n👥 People distribution per frame:")
            for count, frames in sorted(people_distribution.items()):
                percentage = frames / len(all_poses) * 100
                print(f"   {count} people: {frames} frames ({percentage:.1f}%)")
            
            # Quality statistics
            total_keypoints = 0
            valid_keypoints = 0
            
            for frame_poses in all_poses:
                for person_pose in frame_poses:
                    for keypoint in person_pose:
                        total_keypoints += 1
                        if keypoint[2] > self.keypoint_conf_threshold:  # confidence > threshold
                            valid_keypoints += 1
            
            if total_keypoints > 0:
                keypoint_quality = valid_keypoints / total_keypoints * 100
                print(f"\n🎯 Keypoint quality:")
                print(f"   Valid keypoints: {valid_keypoints}/{total_keypoints} ({keypoint_quality:.1f}%)")

    def load_poses_from_json(self, json_path):
        """Load poses từ JSON file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            poses = data['poses']
            video_info = data['video_info']
            
            print(f"📂 Loaded poses from: {json_path}")
            print(f"   Frames: {len(poses)}")
            print(f"   Video: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.2f} FPS")
            
            return poses, video_info
            
        except Exception as e:
            print(f"❌ Error loading poses: {e}")
            raise

def main():
    """Demo sử dụng PoseDetector"""
    print("🎬 YOLO Pose Detector Demo")
    print("=" * 50)
    
    detector = PoseDetector()
    
    # Có thể điều chỉnh thresholds nếu cần
    # detector.set_confidence_thresholds(person_conf=0.25, keypoint_conf=0.2)
    
    # Test với video
    video_path = "res/input/video.mp4"  # Thay đổi path này
    
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return
    try:
        print(f"🚀 Processing video: {video_path}")
        
        # Detect poses từ video
        all_poses, width, height, fps, json_path = detector.detect_poses_from_video(
            video_path=video_path,
            output_path="res/output/output_2d.txt",  # Video với pose overlay
            max_people=10,
            save_frames=False,  # Set True nếu muốn save individual frames
            frame_step=1        # Process every frame
        )
        
        print(f"\n🎉 Processing Complete!")
        print(f"📁 Input video: {video_path}")
        print(f"🎥 Output video: output_with_poses.mp4")
        print(f"📊 Poses data: {json_path}")
        print(f"📐 Video dimensions: {width}x{height}")
        print(f"🎞️  FPS: {fps:.2f}")
        print(f"👥 Total people detected: {sum(len(frame) for frame in all_poses)}")
        
        # Test loading poses back
        print(f"\n🔄 Testing pose loading...")
        loaded_poses, video_info = detector.load_poses_from_json(json_path)
        print(f"✅ Successfully loaded {len(loaded_poses)} frames")
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Tips:")
        print("- Make sure the video file exists and is accessible")
        print("- Check video format (MP4, AVI, MOV supported)")
        print("- Ensure sufficient disk space for output files")
        print("- Try reducing max_people if memory issues occur")

if __name__ == "__main__":
    main()