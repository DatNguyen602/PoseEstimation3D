import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime

class PoseComparison:
    def __init__(self, reference_video_path):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load reference video
        self.reference_video_path = reference_video_path
        self.ref_cap = cv2.VideoCapture(reference_video_path)
        self.ref_fps = self.ref_cap.get(cv2.CAP_PROP_FPS)
        
        # Get video info
        total_frames = int(self.ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Reference video: {total_frames} frames, {self.ref_fps} FPS")
        
        # Video recording
        self.video_writer = None
        self.is_recording = False
        self.output_filename = None
        
    def _extract_keypoints(self, image):
        """Extract pose keypoints from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        keypoints = None
        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y])
            keypoints = np.array(keypoints)
        
        return keypoints, results
    
    def _calculate_score(self, pose1, pose2, threshold=0.1):
        """
        Calculate score based on number of correct keypoints
        
        Scoring method:
        - Count how many keypoints have distance < threshold
        - Score = (correct_keypoints / total_keypoints) * 100%
        
        Threshold guide:
        - 0.05 = Very strict (small movements matter)
        - 0.10 = Medium (default, balanced)
        - 0.15 = Lenient (only major differences matter)
        """
        if pose1 is None or pose2 is None:
            return 0.0, set()
        
        # Reshape to (33, 2) for 33 keypoints with x,y coordinates
        pose1_reshaped = pose1.reshape(-1, 2)
        pose2_reshaped = pose2.reshape(-1, 2)
        
        # Calculate Euclidean distance for each keypoint
        keypoint_distances = np.linalg.norm(pose1_reshaped - pose2_reshaped, axis=1)
        
        # Find correct keypoints (distance <= threshold)
        correct_keypoints = keypoint_distances <= threshold
        
        # Find wrong keypoints (distance > threshold)
        wrong_keypoints = set(np.where(~correct_keypoints)[0].tolist())
        
        # Calculate score based on percentage of correct keypoints
        num_correct = np.sum(correct_keypoints)
        total_keypoints = len(keypoint_distances)
        score = num_correct / total_keypoints  # Returns value between 0 and 1
        
        return score, wrong_keypoints
    
    def _draw_pose(self, frame, results, color=(0, 255, 0), wrong_keypoints=None):
        """Draw pose landmarks on frame with error highlighting"""
        if results.pose_landmarks:
            # Draw landmarks with custom color
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            
            # Draw connections
            connections = self.mp_pose.POSE_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                if start.visibility > 0.5 and end.visibility > 0.5:
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(frame, start_point, end_point, color, 2)
            
            # Draw keypoints
            for idx, landmark in enumerate(landmarks):
                if landmark.visibility > 0.5:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    
                    # Check if this keypoint is wrong
                    if wrong_keypoints is not None and idx in wrong_keypoints:
                        # Draw red circle for wrong keypoints
                        cv2.circle(frame, (x, y), 10, (0, 0, 255), 2)  # Red outer circle
                        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)   # Red filled circle
                    else:
                        # Normal keypoint
                        cv2.circle(frame, (x, y), 4, color, -1)
        
        return frame
    
    def _create_display(self, ref_frame, ref_results, user_frame, user_results, score, wrong_keypoints):
        """Create side-by-side display"""
        # Resize frames
        height = 480
        width = 640
        
        if ref_frame is not None:
            ref_display = cv2.resize(ref_frame, (width, height))
            ref_display = self._draw_pose(ref_display, ref_results, color=(0, 0, 255))  # Red
        else:
            ref_display = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(ref_display, "No Reference", (width//2-100, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        user_display = cv2.resize(user_frame, (width, height))
        # Draw user pose with error highlighting
        user_display = self._draw_pose(user_display, user_results, color=(0, 255, 0), wrong_keypoints=wrong_keypoints)
        
        # Create combined display
        combined = np.hstack([ref_display, user_display])
        
        # Add labels
        cv2.putText(combined, "Sample Pose", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(combined, "Your Pose", (width + 20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add score
        score_text = f"Score: {int(score * 100)}%"
        score_color = (0, 255, 0) if score > 0.7 else (0, 165, 255) if score > 0.4 else (0, 0, 255)
        
        # Score text
        cv2.putText(combined, score_text, (width + 20, height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, score_color, 2)
        
        # Add correct/wrong keypoints count
        total_keypoints = 33
        correct_keypoints = total_keypoints - len(wrong_keypoints)
        count_text = f"Correct: {correct_keypoints}/{total_keypoints}"
        cv2.putText(combined, count_text, (width + 20, height - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return combined, user_display
    
    def _start_recording(self, width, height, fps=30.0):
        """Start recording video"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_filename = f"your_pose_{timestamp}.mp4"
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_filename,
            fourcc,
            fps,
            (width, height)
        )
        
        self.is_recording = True
        print(f"✅ Recording started: {self.output_filename}")
    
    def _stop_recording(self):
        """Stop recording video"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            print(f"✅ Recording saved: {self.output_filename}")
    
    def _write_frame(self, frame, score, wrong_keypoints):
        """Write frame to video with score overlay"""
        if self.is_recording and self.video_writer is not None:
            # Create a copy to add score overlay
            frame_with_score = frame.copy()
            
            # Add score
            score_text = f"Score: {int(score * 100)}%"
            score_color = (0, 255, 0) if score > 0.7 else (0, 165, 255) if score > 0.4 else (0, 0, 255)
            
            # Score background for better visibility
            (text_width, text_height), baseline = cv2.getTextSize(
                score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
            )
            cv2.rectangle(frame_with_score, (10, 10), 
                         (text_width + 30, text_height + 30), (0, 0, 0), -1)
            
            # Score text
            cv2.putText(frame_with_score, score_text, (20, text_height + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, score_color, 3)
            
            # Add correct/wrong count
            total_keypoints = 33
            correct_keypoints = total_keypoints - len(wrong_keypoints)
            count_text = f"Correct: {correct_keypoints}/{total_keypoints}"
            
            cv2.putText(frame_with_score, count_text, (20, text_height + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime('%H:%M:%S')
            cv2.putText(frame_with_score, timestamp, (20, frame_with_score.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            self.video_writer.write(frame_with_score)
    
    def run(self, camera_index=0):
        """Run side-by-side comparison"""
        user_cap = cv2.VideoCapture(camera_index)
        user_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        user_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "="*50)
        print("POSE COMPARISON - WITH VIDEO RECORDING")
        print("="*50)
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Restart reference video")
        print("  'v' - Start/Stop recording your pose")
        print("  SPACE - Pause/Resume")
        print("="*50 + "\n")
        
        paused = False
        score = 0.0
        wrong_keypoints = set()
        
        try:
            while True:
                if not paused:
                    # Read user frame
                    ret_user, user_frame = user_cap.read()
                    if not ret_user:
                        break
                    
                    # Flip for mirror effect
                    user_frame = cv2.flip(user_frame, 1)
                    
                    # Read reference frame
                    ret_ref, ref_frame = self.ref_cap.read()
                    if not ret_ref:
                        # Restart reference video
                        self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_ref, ref_frame = self.ref_cap.read()
                    
                    # Extract poses
                    user_keypoints, user_results = self._extract_keypoints(user_frame)
                    
                    ref_keypoints, ref_results = None, None
                    if ret_ref:
                        ref_keypoints, ref_results = self._extract_keypoints(ref_frame)
                    
                    # Calculate score and find wrong keypoints
                    score, wrong_keypoints = self._calculate_score(user_keypoints, ref_keypoints)
                
                # Create display (use current score even if paused)
                display, user_display = self._create_display(
                    ref_frame if ret_ref else None, ref_results, 
                    user_frame, user_results, score, wrong_keypoints
                )
                
                # Add recording indicator
                if self.is_recording:
                    cv2.circle(display, (10, 10), 10, (0, 0, 255), -1)  # Red dot
                
                # Write frame if recording (only user pose side)
                if self.is_recording and not paused:
                    self._write_frame(user_display, score, wrong_keypoints)
                
                # Show display
                cv2.imshow('Pose Comparison - Reference vs Your Pose', display)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Restart reference video
                    self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print("Reference video restarted")
                elif key == ord('v'):
                    # Toggle recording
                    if not self.is_recording:
                        # Get dimensions from user display
                        height, width = user_display.shape[:2]
                        self._start_recording(width, height, fps=30.0)
                    else:
                        self._stop_recording()
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")
        
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            # Stop recording if still active
            if self.is_recording:
                self._stop_recording()
            user_cap.release()
            self.ref_cap.release()
            cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    # Replace with your reference video path
    reference_video = "res/input/DAN-DO.mp4"
    
    comparison = PoseComparison(reference_video)
    comparison.run(camera_index=0)