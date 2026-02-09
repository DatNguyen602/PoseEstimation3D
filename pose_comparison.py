import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import os
import queue

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
        self.ref_cap = cv2.VideoCapture(reference_video_path) # duong dan video 
        self.ref_fps = self.ref_cap.get(cv2.CAP_PROP_FPS)
        
        # Get video info
        total_frames = int(self.ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
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
    
    def _calculate_score(self, pose1, pose2, threshold=0.07):
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
    
    def _draw_pose(self, frame, main_results, wrong_keypoints=None, ghost_results=None):
        """
        Draws pose skeletons with advanced visualization for errors.
        - Draws a 'ghost' of the reference pose.
        - Draws the user's pose with limbs colored by accuracy (green/red).
        """
        h, w, _ = frame.shape

        # 1. Draw the reference pose "ghost" first (if provided)
        if ghost_results and ghost_results.pose_landmarks:
            ghost_color = (220, 220, 220)  # Light grey for the ghost
            ghost_landmarks = ghost_results.pose_landmarks.landmark
            
            # Create a transparent overlay for the ghost
            overlay = frame.copy()
            alpha = 0.4 # Transparency factor

            for connection in self.mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start = ghost_landmarks[start_idx]
                end = ghost_landmarks[end_idx]
                
                if start.visibility > 0.5 and end.visibility > 0.5:
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(overlay, start_point, end_point, ghost_color, 2)
            
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


        # 2. Draw the user's pose with error highlighting
        if main_results and main_results.pose_landmarks:
            main_landmarks = main_results.pose_landmarks.landmark
            
            # Draw connections with color-coding
            for connection in self.mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                
                # If either point in the connection is wrong, the limb is wrong
                is_wrong = wrong_keypoints is not None and (start_idx in wrong_keypoints or end_idx in wrong_keypoints)
                limb_color = (0, 0, 255) if is_wrong else (0, 255, 0)  # Red if wrong, Green if correct
                
                start = main_landmarks[start_idx]
                end = main_landmarks[end_idx]
                
                if start.visibility > 0.5 and end.visibility > 0.5:
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(frame, start_point, end_point, limb_color, 2)
            
            # Draw keypoints on top
            for idx, landmark in enumerate(main_landmarks):
                if landmark.visibility > 0.5:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    is_wrong = wrong_keypoints is not None and idx in wrong_keypoints
                    point_color = (0, 0, 255) if is_wrong else (0, 255, 0)
                    cv2.circle(frame, (x, y), 4, point_color, -1)
                    
        return frame

    def _create_display(self, ref_frame, ref_results, user_frame, user_results, score, wrong_keypoints):
        """Create side-by-side display with advanced visualization."""
        height, width = 480, 640
        
        # --- Reference Pane ---
        # Simple display of the reference pose
        if ref_frame is not None:
            ref_display = cv2.resize(ref_frame, (width, height))
            if ref_results and ref_results.pose_landmarks:
                 self.mp_drawing.draw_landmarks(
                    ref_display, ref_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        else:
            ref_display = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(ref_display, "No Reference", (width//2-100, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # --- User Pane ---
        # Advanced display with ghost and error coloring
        user_display = cv2.resize(user_frame, (width, height))
        user_display = self._draw_pose(
            user_display, 
            main_results=user_results, 
            wrong_keypoints=wrong_keypoints, 
            ghost_results=ref_results
        )
        
        # Create combined display
        combined = np.hstack([ref_display, user_display])
        
        # Add labels
        cv2.putText(combined, "Sample Pose", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Your Pose", (width + 20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add score
        score_text = f"Score: {int(score * 100)}%"
        score_color = (0, 255, 0) if score > 0.7 else (0, 165, 255) if score > 0.4 else (0, 0, 255)
        
        cv2.putText(combined, score_text, (width + 20, height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, score_color, 2)
        
        # Add correct/wrong keypoints count
        if wrong_keypoints is not None:
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
    
    def process_video_files(self, user_video_path: str, output_path: str, progress_queue: 'queue.Queue'):
        """
        Compares a user's video against the reference video and saves a side-by-side comparison video.
        Reports progress via a queue.
        """
        user_cap = cv2.VideoCapture(user_video_path)
        if not user_cap.isOpened():
            raise ValueError(f"Could not open user video: {user_video_path}")

        # Get video properties
        ref_frame_count = int(self.ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        user_frame_count = int(user_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = min(ref_frame_count, user_frame_count)

        if total_frames == 0:
            raise ValueError("One of the videos has 0 frames.")

        user_fps = user_cap.get(cv2.CAP_PROP_FPS)
        fps = min(self.ref_fps, user_fps) if self.ref_fps > 0 and user_fps > 0 else 30

        # For the output video, we'll use the standard display size from _create_display
        output_width = 640 * 2  # Side-by-side
        output_height = 480

        # Create VideoWriter for the output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

        self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind reference video

        try:
            for i in range(total_frames):
                # Read frames
                ret_ref, ref_frame = self.ref_cap.read()
                ret_user, user_frame = user_cap.read()

                if not ret_ref or not ret_user:
                    break

                # Process frames
                user_keypoints, user_results = self._extract_keypoints(user_frame)
                ref_keypoints, ref_results = self._extract_keypoints(ref_frame)

                score, wrong_keypoints = self._calculate_score(user_keypoints, ref_keypoints)

                # Create the combined display frame
                display_frame, _ = self._create_display(
                    ref_frame, ref_results,
                    user_frame, user_results, score, wrong_keypoints
                )

                # Write frame to output video
                video_writer.write(display_frame)

                # Report detailed progress (every 10 frames for better UX)
                if i % 10 == 0:
                    progress_percentage = 20 + int((i / total_frames) * 70)  # 20-90% range for frame processing
                    progress_message = f"Processed frame {i+1}/{total_frames} ({progress_percentage}%)"
                    progress_queue.put({"type": "progress", "step": "processing_frames", "message": progress_message, "percentage": progress_percentage})
        
        except Exception as e:
            import traceback
            progress_queue.put({"type": "error", "data": f"ERROR: {str(e)}\n{traceback.format_exc()}"})

        finally:
            # Release all resources
            user_cap.release()
            self.ref_cap.release()
            video_writer.release()
            progress_queue.put({"type": "progress", "step": "saving_video", "message": "Saving comparison video... (will be preserved for viewing)", "percentage": 95})
            progress_queue.put({"type": "progress", "step": "completed", "message": f"✅ Comparison video saved to {output_path} and preserved for viewing", "percentage": 100})

    def annotate_video(self, raw_user_video_path: str, annotated_output_path: str):
        """
        Takes a raw user video, compares it against the reference, and creates a new
        video with the user's skeleton, errors, and a reference 'ghost' drawn on it.
        """
        user_cap = cv2.VideoCapture(raw_user_video_path)
        if not user_cap.isOpened():
            raise ValueError(f"Could not open raw user video: {raw_user_video_path}")

        # Get video properties
        ref_frame_count = int(self.ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        user_frame_count = int(user_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = min(ref_frame_count, user_frame_count)

        if total_frames == 0:
            user_cap.release()
            raise ValueError("Input video for annotation has 0 frames.")

        fps = user_cap.get(cv2.CAP_PROP_FPS)
        width = int(user_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(user_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create VideoWriter for the annotated output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(annotated_output_path, fourcc, fps, (width, height))

        self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind reference video

        try:
            for i in range(total_frames):
                ret_ref, ref_frame = self.ref_cap.read()
                ret_user, user_frame = user_cap.read()

                if not ret_ref or not ret_user:
                    break

                # Perform comparison to get data
                user_keypoints, user_results = self._extract_keypoints(user_frame)
                ref_keypoints, ref_results = self._extract_keypoints(ref_frame)
                _, wrong_keypoints = self._calculate_score(user_keypoints, ref_keypoints)

                # Draw the advanced visualization on the user frame
                annotated_frame = self._draw_pose(
                    user_frame,
                    main_results=user_results,
                    wrong_keypoints=wrong_keypoints,
                    ghost_results=ref_results
                )
                
                # Write the annotated frame
                video_writer.write(annotated_frame)
        finally:
            user_cap.release()
            video_writer.release()
            print(f"✅ Annotation complete. Final video saved to: {annotated_output_path} (preserved for viewing)")

    #
    def run(self, camera_index=0):
        """Run side-by-side comparison"""
        user_cap = cv2.VideoCapture(camera_index)#lấy cam realtime 
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
                    #lấy khung hình từ camera
                    ret_user, user_frame = user_cap.read()
                    if not ret_user:
                        break
                    
                    # Flip for mirror effect
                    user_frame = cv2.flip(user_frame, 1)
                    
                    # Read reference frame
                    #lấy khung hình từ video
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
    reference_video = "res/input/video.mp4"
    
    comparison = PoseComparison(reference_video)#truyền video đâu vào 
    comparison.run(camera_index=0)


class LiveComparisonSession:
    """Manages a single real-time comparison session, designed for WebSocket usage."""

    def __init__(self, reference_video_path: str, output_dir: str):
        self.comparison = PoseComparison(reference_video_path)
        self.output_path = None
        self._initialize_recording(output_dir)

    def _initialize_recording(self, output_dir: str):
        """Starts the recording process for the user's performance."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_path = os.path.join(output_dir, f"live_session_{timestamp}.mp4")
        
        # Assuming a standard webcam resolution for the output
        # The actual frame size will be used when writing the first frame.
        # Placeholder dimensions, will be updated.
        self.width = 640 
        self.height = 480
        fps = 20 # A reasonable default for webcam streams

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (self.width, self.height))
        self.is_recording = True
        print(f"✅ Live session recording started: {self.output_path}")

    def process_frame(self, user_frame_bytes: bytes) -> dict:
        """Processes a single frame from the user, compares it, and returns the result."""
        # Decode user frame
        nparr = np.frombuffer(user_frame_bytes, np.uint8)
        user_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if user_frame is None:
            return {"error": "Invalid frame received."}

        # Update recording dimensions if this is the first frame
        if self.video_writer is not None and (self.height, self.width) != user_frame.shape[:2]:
            self.height, self.width, _ = user_frame.shape
            fps = self.video_writer.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer.release()
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (self.width, self.height))

        # Read corresponding reference frame
        ret_ref, ref_frame = self.comparison.ref_cap.read()
        if not ret_ref:
            # If reference video ends, loop it
            self.comparison.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_ref, ref_frame = self.comparison.ref_cap.read()
            if not ret_ref:
                return {"error": "Could not read reference video."}

        # Perform comparison
        user_keypoints, user_results = self.comparison._extract_keypoints(user_frame)
        ref_keypoints, _ = self.comparison._extract_keypoints(ref_frame)
        score, wrong_keypoints = self.comparison._calculate_score(user_keypoints, ref_keypoints)

        # Write plain user's frame to the recording for post-processing later
        if self.is_recording:
            self.video_writer.write(user_frame)

        # Convert numpy arrays to lists for JSON serialization
        user_kps_list = user_keypoints.tolist() if user_keypoints is not None else []
        ref_kps_list = ref_keypoints.tolist() if ref_keypoints is not None else []

        return {
            "score": score,
            "wrong_keypoints": list(wrong_keypoints),
            "user_keypoints": user_kps_list,
            "ref_keypoints": ref_kps_list
        }

    def close(self) -> str:
        """Stops recording, releases all resources, and returns the output path."""
        if self.is_recording:
            self.video_writer.release()
            self.is_recording = False
            print(f"✅ Live session recording saved: {self.output_path}")
        
        if self.comparison.ref_cap.isOpened():
            self.comparison.ref_cap.release()
            
        return self.output_path