import cv2
import numpy as np
import json
from ultralytics import YOLO
import mediapipe as mp
import os
import time
from pathlib import Path

class PoseDetectorWithHands:
    def __init__(self, yolo_model_path='yolo11n-pose.pt'):
        """
        Khởi tạo detector kết hợp YOLO + MediaPipe
        
        Args:
            yolo_model_path: Path đến YOLO pose model
        """
        print(f"🤖 Initializing YOLO + MediaPipe Detector...")
        
        # Initialize YOLO for body poses
        try:
            self.yolo_model = YOLO(yolo_model_path)
            print(f"✅ YOLO loaded: {yolo_model_path}")
        except Exception as e:
            print(f"❌ Error loading YOLO: {e}")
            raise
        
        # Initialize MediaPipe for hands
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=4,  # Max 2 hands per person, up to 2 people
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
            print(f"✅ MediaPipe Hands loaded")
        except Exception as e:
            print(f"❌ Error loading MediaPipe: {e}")
            self.hands = None
        
        # Confidence thresholds
        self.person_conf_threshold = 0.45
        self.keypoint_conf_threshold = 0.3
        self.hand_conf_threshold = 0.5
        
        self.setup_keypoint_structure()
        
        # Colors for visualization
        self.person_colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128)
        ]
        
        print(f"🎯 Detection Configuration:")
        print(f"   Body keypoints: {self.num_body_keypoints} (YOLO)")
        print(f"   Hand keypoints: {self.num_hand_keypoints} (MediaPipe)")
        print(f"   Total keypoints: {self.total_keypoints}")
    
    def setup_keypoint_structure(self):
        """Setup keypoint structure"""
        
        # YOLO Body keypoints (17)
        self.body_keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # MediaPipe Hand keypoints (21 per hand, 42 total)
        hand_landmark_names = [
            'wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
            'index_mcp', 'index_pip', 'index_dip', 'index_tip',
            'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip', 
            'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
            'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
        ]
        
        self.left_hand_names = [f'left_hand_{name}' for name in hand_landmark_names]
        self.right_hand_names = [f'right_hand_{name}' for name in hand_landmark_names]
        
        # Combined keypoint names
        self.all_keypoint_names = (self.body_keypoint_names + 
                                  self.left_hand_names + 
                                  self.right_hand_names)
        
        self.num_body_keypoints = 17
        self.num_hand_keypoints = 42  # 21 per hand × 2 hands
        self.total_keypoints = 59  # 17 + 42
        
        # Keypoint indices
        self.body_indices = list(range(17))
        self.left_hand_indices = list(range(17, 38))   # 17-37
        self.right_hand_indices = list(range(38, 59))  # 38-58
        
        # YOLO skeleton connections
        self.body_skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms  
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # MediaPipe hand connections
        self.hand_connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger  
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm connections
            (5, 9), (9, 13), (13, 17)
        ]
    
    def detect_poses_from_video(self, video_path, output_video_path=None, max_people=2,
                               detect_hands=True, save_frames=False, frame_step=1):
        """
        Detect body + hand poses từ video
        
        Args:
            video_path: Input video path
            output_video_path: Output video path với pose overlay
            max_people: Max số người (recommend ≤ 2 for hand detection performance)
            detect_hands: Có detect hands không
            save_frames: Save individual frames
            frame_step: Frame step
            
        Returns:
            tuple: (all_poses, width, height, fps, json_path)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"📹 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
        print(f"🎯 Detection: Body (YOLO) + {'Hands (MediaPipe)' if detect_hands else 'No hands'}")
        
        # Initialize outputs
        all_poses = []
        frame_idx = 0
        processed_frames = 0
        
        # Video writer
        video_writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps/frame_step, (width, height))
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Frame stepping
            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue
            
            # Detect body poses with YOLO
            yolo_results = self.yolo_model(frame, verbose=False, 
                                          conf=self.person_conf_threshold, 
                                          iou=0.5, max_det=max_people)
            
            # Detect hands with MediaPipe (if enabled)
            hand_results = None
            if detect_hands and self.hands:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_results = self.hands.process(rgb_frame)
            
            # Combine detections
            frame_poses = self._combine_body_and_hands(
                yolo_results, hand_results, frame, max_people, detect_hands
            )
            
            all_poses.append(frame_poses)
            
            # Visualization
            if output_video_path and video_writer:
                annotated_frame = self._draw_combined_poses(
                    frame.copy(), yolo_results, hand_results, max_people, detect_hands
                )
                video_writer.write(annotated_frame)
            
            processed_frames += 1
            frame_idx += 1
            
            # Progress
            if processed_frames % 50 == 0:
                elapsed = time.time() - start_time
                speed = processed_frames / elapsed
                eta = (total_frames/frame_step - processed_frames) / speed if speed > 0 else 0
                
                detected_people = len(frame_poses)
                print(f"📈 {processed_frames}/{total_frames//frame_step} frames "
                      f"({processed_frames/(total_frames//frame_step)*100:.1f}%) | "
                      f"{speed:.1f} FPS | People: {detected_people} | ETA: {eta:.1f}s")
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        
        # Save results
        json_path = self._save_combined_poses_to_json(
            all_poses, video_path, width, height, fps/frame_step, detect_hands
        )
        
        # Statistics
        total_time = time.time() - start_time
        self._print_detection_statistics(all_poses, processed_frames, total_time, detect_hands)
        
        return all_poses, width, height, fps/frame_step, json_path
    
    def _combine_body_and_hands(self, yolo_results, hand_results, frame, max_people, detect_hands):
        """Combine YOLO body detection với MediaPipe hand detection"""
        frame_poses = []
        
        # Extract body poses từ YOLO
        body_poses = self._extract_body_poses(yolo_results, max_people)
        
        if not body_poses:
            return frame_poses
        
        # Extract hand poses từ MediaPipe (nếu có)
        hand_landmarks_list = []
        if detect_hands and hand_results and hand_results.multi_hand_landmarks:
            hand_landmarks_list = hand_results.multi_hand_landmarks
        
        # Combine cho từng người
        for person_idx, body_pose in enumerate(body_poses):
            # Initialize combined pose: 59 keypoints (17 body + 42 hands)
            combined_pose = np.zeros((59, 3))
            
            # Copy body keypoints (0-16)
            combined_pose[:17] = body_pose
            
            # Add hand keypoints nếu available
            if detect_hands:
                left_hand, right_hand = self._match_hands_to_person(
                    body_pose, hand_landmarks_list, frame.shape[:2]
                )
                
                if left_hand is not None:
                    combined_pose[17:38] = left_hand  # Left hand indices 17-37
                
                if right_hand is not None:
                    combined_pose[38:59] = right_hand  # Right hand indices 38-58
            
            frame_poses.append(combined_pose.tolist())
        
        return frame_poses
    
    def _extract_body_poses(self, yolo_results, max_people):
        """Extract body poses từ YOLO results"""
        body_poses = []
        
        if not yolo_results or len(yolo_results) == 0:
            return body_poses
        
        keypoints = yolo_results[0].keypoints
        if keypoints is None or len(keypoints.data) == 0:
            return body_poses
        
        for person_idx, person_keypoints in enumerate(keypoints.data):
            if person_idx >= max_people:
                break
            
            # Extract 17 body keypoints
            body_pose = np.zeros((17, 3))
            
            for i in range(min(17, len(person_keypoints))):
                kpt = person_keypoints[i]
                body_pose[i] = [
                    self._extract_value(kpt[0]),
                    self._extract_value(kpt[1]),
                    self._extract_value(kpt[2])
                ]
            
            body_poses.append(body_pose)
        
        return body_poses
    
    def _match_hands_to_person(self, body_pose, hand_landmarks_list, image_shape):
        """Match detected hands với specific person dựa trên wrist positions"""
        if not hand_landmarks_list:
            return None, None
        
        height, width = image_shape
        
        # Get wrist positions từ body pose
        left_wrist = body_pose[9]   # left_wrist index
        right_wrist = body_pose[10] # right_wrist index
        
        # Convert hand landmarks to pixel coordinates
        detected_hands = []
        for hand_landmarks in hand_landmarks_list:
            hand_points = []
            for landmark in hand_landmarks.landmark:
                x = landmark.x * width
                y = landmark.y * height
                z = landmark.z  # relative depth
                hand_points.append([x, y, 0.8])  # confidence = 0.8
            
            detected_hands.append(np.array(hand_points))
        
        # Match hands to wrists
        left_hand = None
        right_hand = None
        
        if len(detected_hands) > 0:
            # Find closest hand to each wrist
            for hand_points in detected_hands:
                hand_wrist = hand_points[0]  # Wrist is first landmark
                
                # Calculate distances to body wrists
                left_dist = np.inf
                right_dist = np.inf
                
                if left_wrist[2] > self.keypoint_conf_threshold:  # Left wrist detected
                    left_dist = np.linalg.norm(hand_wrist[:2] - left_wrist[:2])
                
                if right_wrist[2] > self.keypoint_conf_threshold:  # Right wrist detected
                    right_dist = np.linalg.norm(hand_wrist[:2] - right_wrist[:2])
                
                # Assign to closest wrist (with distance threshold)
                max_distance = 100  # pixels
                
                if left_dist < right_dist and left_dist < max_distance and left_hand is None:
                    left_hand = hand_points
                elif right_dist < max_distance and right_hand is None:
                    right_hand = hand_points
        
        return left_hand, right_hand
    
    def _extract_value(self, tensor_value):
        """Extract value từ tensor"""
        if hasattr(tensor_value, 'cpu'):
            return float(tensor_value.cpu().numpy())
        elif hasattr(tensor_value, 'item'):
            return float(tensor_value.item())
        else:
            return float(tensor_value)
    
    def _draw_combined_poses(self, frame, yolo_results, hand_results, max_people, detect_hands):
        """Draw combined body + hand poses"""
        
        # Draw body poses
        if yolo_results and len(yolo_results) > 0:
            keypoints = yolo_results[0].keypoints
            
            if keypoints is not None and len(keypoints.data) > 0:
                for person_idx, person_keypoints in enumerate(keypoints.data):
                    if person_idx >= max_people:
                        break
                    
                    color = self.person_colors[person_idx % len(self.person_colors)]
                    
                    # Draw body keypoints
                    self._draw_body_keypoints(frame, person_keypoints, color)
                    self._draw_body_skeleton(frame, person_keypoints, color)
                    
                    # Draw person ID
                    self._draw_person_id(frame, person_keypoints, person_idx, color)
        
        # Draw hands with same colors as body
        if detect_hands and hand_results and hand_results.multi_hand_landmarks:
            self._draw_hands_with_body_colors(frame, hand_results, yolo_results, max_people)
        
        return frame
    
    def _draw_body_keypoints(self, frame, person_keypoints, color):
        """Draw body keypoints"""
        for i in range(min(17, len(person_keypoints))):
            kpt = person_keypoints[i]
            conf = self._extract_value(kpt[2])
            
            if conf > self.keypoint_conf_threshold:
                x = int(self._extract_value(kpt[0]))
                y = int(self._extract_value(kpt[1]))
                cv2.circle(frame, (x, y), 4, color, -1)
                cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)
    
    def _draw_body_skeleton(self, frame, person_keypoints, color):
        """Draw body skeleton"""
        for start_idx, end_idx in self.body_skeleton:
            if start_idx < len(person_keypoints) and end_idx < len(person_keypoints):
                start_conf = self._extract_value(person_keypoints[start_idx][2])
                end_conf = self._extract_value(person_keypoints[end_idx][2])
                
                if (start_conf > self.keypoint_conf_threshold and 
                    end_conf > self.keypoint_conf_threshold):
                    
                    start_x = int(self._extract_value(person_keypoints[start_idx][0]))
                    start_y = int(self._extract_value(person_keypoints[start_idx][1]))
                    end_x = int(self._extract_value(person_keypoints[end_idx][0]))
                    end_y = int(self._extract_value(person_keypoints[end_idx][1]))
                    
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)
    
    def _draw_hands_with_body_colors(self, frame, hand_results, yolo_results, max_people):
        """Draw hand landmarks và connections với màu giống body"""
        if not hand_results.multi_hand_landmarks:
            return
        
        height, width = frame.shape[:2]
        
        # Get body wrist positions for matching
        body_wrists = []
        if yolo_results and len(yolo_results) > 0:
            keypoints = yolo_results[0].keypoints
            if keypoints is not None and len(keypoints.data) > 0:
                for person_idx, person_keypoints in enumerate(keypoints.data):
                    if person_idx >= max_people:
                        break
                    
                    color = self.person_colors[person_idx % len(self.person_colors)]
                    
                    # Get wrist positions
                    left_wrist = None
                    right_wrist = None
                    
                    if len(person_keypoints) > 9:  # left_wrist index 9
                        left_wrist_kpt = person_keypoints[9]
                        left_wrist_conf = self._extract_value(left_wrist_kpt[2])
                        if left_wrist_conf > self.keypoint_conf_threshold:
                            left_wrist = (
                                int(self._extract_value(left_wrist_kpt[0])),
                                int(self._extract_value(left_wrist_kpt[1]))
                            )
                    
                    if len(person_keypoints) > 10:  # right_wrist index 10
                        right_wrist_kpt = person_keypoints[10]
                        right_wrist_conf = self._extract_value(right_wrist_kpt[2])
                        if right_wrist_conf > self.keypoint_conf_threshold:
                            right_wrist = (
                                int(self._extract_value(right_wrist_kpt[0])),
                                int(self._extract_value(right_wrist_kpt[1]))
                            )
                    
                    body_wrists.append({
                        'person_idx': person_idx,
                        'color': color,
                        'left_wrist': left_wrist,
                        'right_wrist': right_wrist
                    })
        
        # Draw each hand with matching body color
        for hand_landmarks in hand_results.multi_hand_landmarks:
            hand_wrist_landmark = hand_landmarks.landmark[0]  # Wrist is first landmark
            hand_wrist_x = int(hand_wrist_landmark.x * width)
            hand_wrist_y = int(hand_wrist_landmark.y * height)
            
            # Find closest body wrist
            best_match = None
            min_distance = float('inf')
            max_distance = 100  # pixels
            
            for body_data in body_wrists:
                # Check left wrist
                if body_data['left_wrist']:
                    dist = np.linalg.norm(
                        np.array([hand_wrist_x, hand_wrist_y]) - 
                        np.array(body_data['left_wrist'])
                    )
                    if dist < min_distance and dist < max_distance:
                        min_distance = dist
                        best_match = body_data
                
                # Check right wrist
                if body_data['right_wrist']:
                    dist = np.linalg.norm(
                        np.array([hand_wrist_x, hand_wrist_y]) - 
                        np.array(body_data['right_wrist'])
                    )
                    if dist < min_distance and dist < max_distance:
                        min_distance = dist
                        best_match = body_data
            
            # Use matched color or default color
            hand_color = best_match['color'] if best_match else (0, 255, 255)
            
            # Draw hand connections với màu matched
            for start_idx, end_idx in self.hand_connections:
                start_landmark = hand_landmarks.landmark[start_idx]
                end_landmark = hand_landmarks.landmark[end_idx]
                
                start_x = int(start_landmark.x * width)
                start_y = int(start_landmark.y * height)
                end_x = int(end_landmark.x * width)
                end_y = int(end_landmark.y * height)
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), hand_color, 2)
            
            # Draw hand keypoints với màu matched
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 3, hand_color, -1)
                cv2.circle(frame, (x, y), 4, (255, 255, 255), 1)
    
    def _draw_person_id(self, frame, person_keypoints, person_idx, color):
        """Draw person ID"""
        if len(person_keypoints) > 0:
            nose_kpt = person_keypoints[0]
            nose_conf = self._extract_value(nose_kpt[2])
            
            if nose_conf > self.keypoint_conf_threshold:
                x = int(self._extract_value(nose_kpt[0]))
                y = int(self._extract_value(nose_kpt[1]))
                
                text = f"Person {person_idx}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                cv2.rectangle(frame, (x-5, y-25), (x+text_size[0]+5, y-5), color, -1)
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 2)
    
    def _save_combined_poses_to_json(self, all_poses, video_path, width, height, fps, detect_hands):
        """Save combined poses to JSON"""
        pose_data = {
            'poses': all_poses,
            'video_info': {
                'source_video': os.path.basename(video_path),
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': len(all_poses)
            },
            'detection_settings': {
                'person_conf_threshold': self.person_conf_threshold,
                'keypoint_conf_threshold': self.keypoint_conf_threshold,
                'hand_conf_threshold': self.hand_conf_threshold,
                'detect_hands': detect_hands,
                'body_model': 'YOLO11n-pose',
                'hand_model': 'MediaPipe Hands' if detect_hands else None
            },
            'keypoint_format': {
                'format_name': 'YOLO_Body_MediaPipe_Hands',
                'total_keypoints': self.total_keypoints,
                'body_keypoints': self.num_body_keypoints,
                'hand_keypoints': self.num_hand_keypoints if detect_hands else 0,
                'coordinate_format': 'pixel_coordinates',
                'data_structure': 'frames -> people -> keypoints -> [x, y, confidence]',
                'keypoint_names': self.all_keypoint_names
            },
            'keypoint_indices': {
                'body_range': [0, 16],              # YOLO body keypoints
                'left_hand_range': [17, 37],        # MediaPipe left hand
                'right_hand_range': [38, 58]        # MediaPipe right hand
            }
        }
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        json_path = f"{base_name}_body_hands_poses_2d.json"
        
        with open(json_path, 'w') as f:
            json.dump(pose_data, f, indent=2)
        
        print(f"💾 Saved body+hands poses to: {json_path}")
        return json_path
    
    def _print_detection_statistics(self, all_poses, total_frames, total_time, detect_hands):
        """Print detection statistics"""
        if not all_poses:
            print("⚠️ No poses detected!")
            return
        
        total_people = sum(len(frame_poses) for frame_poses in all_poses)
        avg_people_per_frame = total_people / len(all_poses)
        frames_with_detections = sum(1 for frame_poses in all_poses if len(frame_poses) > 0)
        detection_rate = (frames_with_detections / len(all_poses)) * 100
        
        # Count valid hand detections
        total_hands = 0
        if detect_hands:
            for frame_poses in all_poses:
                for person_pose in frame_poses:
                    person_array = np.array(person_pose)
                    # Check left hand (indices 17-38)
                    left_hand_valid = np.sum(person_array[17:38, 2] > 0.1)
                    # Check right hand (indices 38-59) 
                    right_hand_valid = np.sum(person_array[38:59, 2] > 0.1)
                    if left_hand_valid > 5:
                        total_hands += 1
                    if right_hand_valid > 5:
                        total_hands += 1
        
        print(f"\n📊 Detection Statistics:")
        print(f"   Processing time: {total_time:.2f}s")
        print(f"   Average speed: {total_frames/total_time:.1f} FPS")
        print(f"   Total frames: {total_frames}")
        print(f"   Detection rate: {detection_rate:.1f}%")
        print(f"   People detected: {total_people}")
        print(f"   Avg people/frame: {avg_people_per_frame:.2f}")
        if detect_hands:
            print(f"   Valid hands detected: {total_hands}")
        print(f"   Keypoints per person: {self.total_keypoints} "
              f"(Body: {self.num_body_keypoints}, Hands: {self.num_hand_keypoints if detect_hands else 0})")

def main():
    """Demo YOLO + MediaPipe detector"""
    print("🎬 YOLO Body + MediaPipe Hands Detector Demo")
    print("=" * 60)
    
    detector = PoseDetectorWithHands()
    
    video_path = "res/input/DAN-DO.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        print("Please place your video file or update video_path")
        
        # Create sample data for testing
        print(f"\n🔧 Creating sample data...")
        sample_poses = []
        
        # Create 30 frames with 2 people
        for frame_idx in range(30):
            frame_data = []
            
            # Person 1: with hands
            person1_pose = np.zeros((59, 3))
            # Body keypoints (sample)
            person1_pose[:17] = [[100+i*10, 200+i*15, 0.8] for i in range(17)]
            # Left hand keypoints (sample)
            person1_pose[17:38] = [[120+i*5, 180+i*3, 0.7] for i in range(21)]
            # Right hand keypoints (sample)
            person1_pose[38:59] = [[200+i*5, 180+i*3, 0.7] for i in range(21)]
            
            frame_data.append(person1_pose.tolist())
            
            # Person 2: body only
            person2_pose = np.zeros((59, 3))
            person2_pose[:17] = [[300+i*10, 200+i*15, 0.8] for i in range(17)]
            # Hands left as zeros
            
            frame_data.append(person2_pose.tolist())
            sample_poses.append(frame_data)
        
        # Save sample data
        sample_data = {
            'poses': sample_poses,
            'video_info': {'width': 1920, 'height': 1080, 'fps': 30.0},
            'keypoint_format': {'total_keypoints': 59, 'body_keypoints': 17, 'hand_keypoints': 42}
        }
        
        with open('sample_body_hands_poses_2d.json', 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print("✅ Created sample data: sample_body_hands_poses_2d.json")
        return
    
    try:
        print(f"🚀 Processing: {video_path}")
        
        # Detect với hands
        all_poses, width, height, fps, json_path = detector.detect_poses_from_video(
            video_path=video_path,
            output_video_path="output_body_hands_poses.mp4",
            max_people=2,      # Recommend ≤ 2 for hand detection performance
            detect_hands=True, # Enable hand detection
            frame_step=1
        )
        
        print(f"\n🎉 Detection Complete!")
        print(f"📁 Input: {video_path}")
        print(f"🎥 Output video: output_body_hands_poses.mp4")
        print(f"📊 Poses data: {json_path}")
        print(f"👥 Total detections: {sum(len(frame) for frame in all_poses)}")
        print(f"🤚 Keypoints breakdown:")
        print(f"   Body: 17 keypoints (YOLO)")
        print(f"   Hands: 42 keypoints (MediaPipe - 21 per hand)")
        print(f"   Total: 59 keypoints per person")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()