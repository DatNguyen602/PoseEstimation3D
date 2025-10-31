import os
import json
import numpy as np
import torch
import time
from pathlib import Path
from datetime import datetime
from typing import Dict
from videopose3d_model import VideoPose3DPredictor, VIDEOPOSE3D_AVAILABLE

class Pose3DEstimator:
    def __init__(self, checkpoint_path=None, device=None, architecture=None):
        """
        Khởi tạo 3D pose estimator
        
        Args:
            checkpoint_path: Đường dẫn đến pretrained VideoPose3D checkpoint
            device: PyTorch device (auto-detect if None)
            architecture: Model architecture (default: [3,3,3,3,3])
        """
        print(f"🎯 Initializing 3D Pose Estimator...")
        
        if not VIDEOPOSE3D_AVAILABLE:
            raise RuntimeError("VideoPose3D not available! Please setup VideoPose3D repository first.")
        
        # Initialize VideoPose3D predictor
        self.predictor = VideoPose3DPredictor(
            checkpoint_path=checkpoint_path,
            device=device,
            architecture=architecture
        )
        
        # Human3.6M joint names (17 joints)
        self.joint_names = [
            'hip',           # 0  - Root joint
            'right_hip',     # 1  
            'right_knee',    # 2
            'right_foot',    # 3
            'left_hip',      # 4
            'left_knee',     # 5
            'left_foot',     # 6
            'spine',         # 7
            'thorax',        # 8
            'neck',          # 9
            'head',          # 10
            'left_shoulder', # 11
            'left_elbow',    # 12
            'left_wrist',    # 13
            'right_shoulder',# 14
            'right_elbow',   # 15
            'right_wrist'    # 16
        ]
        
        # Skeleton connections for visualization
        self.skeleton_connections = [
            (0, 1), (1, 2), (2, 3),  # Right leg
            (0, 4), (4, 5), (5, 6),  # Left leg
            (0, 7), (7, 8), (8, 9), (9, 10),  # Spine
            (8, 11), (11, 12), (12, 13),  # Left arm
            (8, 14), (14, 15), (15, 16),  # Right arm
        ]
        
        # Processing settings - Điều chỉnh cho tỷ lệ thành công tối đa
        self.min_frames_threshold = 10  # Giảm xuống 1 frame duy nhất
        self.confidence_threshold = 0.1  # Giảm xuống rất thấp
        self.min_valid_keypoints = 5     # Chỉ cần 1 keypoint hợp lệ
        
        print(f"✅ 3D Pose Estimator ready!")
        print(f"   Minimum frames per person: {self.min_frames_threshold}")
        print(f"   Confidence threshold: {self.confidence_threshold}")
        print(f"   Minimum valid keypoints: {self.min_valid_keypoints}")
    
    def process_2d_poses_file(self, poses_2d_file, output_prefix="output_3d", 
                             enable_filtering=True, enable_smoothing=False):
        """
        Xử lý file 2D poses để tạo 3D poses cho nhiều người
        
        Args:
            poses_2d_file: Đường dẫn đến file JSON chứa 2D poses
            output_prefix: Prefix cho output files
            enable_filtering: Có filter low-quality poses không
            enable_smoothing: Có apply temporal smoothing không
            
        Returns:
            tuple: (all_people_poses_3d, metadata)
        """
        print(f"\n🚀 Starting 3D pose estimation pipeline...")
        start_time = time.time()
        
        # Load và validate 2D poses
        poses_2d_data = self._load_and_validate_2d_poses(poses_2d_file)
        
        poses_2d = poses_2d_data['poses']
        width = poses_2d_data['video_info']['width']
        height = poses_2d_data['video_info']['height']
        fps = poses_2d_data['video_info'].get('fps', 30.0)
        
        print(f"📊 Input data:")
        print(f"   Video: {width}x{height} @ {fps:.2f} FPS")
        print(f"   Total frames: {len(poses_2d)}")
        
        # Analyze multi-person structure
        people_analysis = self._analyze_people_structure(poses_2d)
        
        # Process each person
        all_people_poses_3d, processing_stats = self._process_multiple_people(
            poses_2d, width, height, people_analysis, 
            enable_filtering, enable_smoothing
        )
        
        # Create metadata
        total_time = time.time() - start_time
        metadata = self._create_metadata(
            poses_2d_file, poses_2d_data, processing_stats, total_time
        )
        
        # Save results
        output_paths = self._save_3d_poses(
            all_people_poses_3d, metadata, output_prefix
        )
        
        # Print summary
        self._print_processing_summary(metadata, output_paths, total_time)
        
        valid_poses = [poses for poses in all_people_poses_3d if poses is not None]
        return valid_poses, metadata
    
    def _load_and_validate_2d_poses(self, poses_2d_file):
        """Load và validate 2D poses file"""
        if not os.path.exists(poses_2d_file):
            raise FileNotFoundError(f"2D poses file not found: {poses_2d_file}")
        
        print(f"📂 Loading 2D poses from: {poses_2d_file}")
        
        try:
            with open(poses_2d_file, 'r') as f:
                data = json.load(f)
            
            # Validate data structure
            required_keys = ['poses', 'video_info']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key: {key}")
            
            # Validate video info
            video_info = data['video_info']
            required_video_keys = ['width', 'height']
            for key in required_video_keys:
                if key not in video_info:
                    raise ValueError(f"Missing video info key: {key}")
            
            poses = data['poses']
            if not isinstance(poses, list) or len(poses) == 0:
                raise ValueError("Invalid or empty poses data")
            
            print(f"✅ Loaded and validated 2D poses")
            print(f"   Format: {data.get('data_format', {}).get('keypoint_format', 'Unknown')}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error loading 2D poses: {e}")
            raise
    
    def _analyze_people_structure(self, poses_2d):
        """Phân tích cấu trúc multi-person trong data"""
        max_people = max(len(frame) for frame in poses_2d) if poses_2d else 0
        
        people_analysis = []
        
        for person_idx in range(max_people):
            # Count appearances và valid frames
            appearances = 0
            valid_frames = 0
            quality_scores = []
            
            for frame_idx, frame_poses in enumerate(poses_2d):
                if person_idx < len(frame_poses):
                    appearances += 1
                    person_pose = np.array(frame_poses[person_idx])
                    
                    # Check frame quality
                    if person_pose.shape[1] >= 3:  # Has confidence
                        valid_keypoints = np.sum(person_pose[:, 2] > self.confidence_threshold)
                    else:
                        valid_keypoints = np.sum(np.any(person_pose[:, :2] > 0, axis=1))
                    
                    if valid_keypoints >= self.min_valid_keypoints:
                        valid_frames += 1
                        quality_scores.append(valid_keypoints / len(person_pose))
            
            analysis = {
                'person_id': person_idx,
                'total_appearances': appearances,
                'valid_frames': valid_frames,
                'appearance_rate': appearances / len(poses_2d) if poses_2d else 0,
                'valid_rate': valid_frames / appearances if appearances > 0 else 0,
                'avg_quality': np.mean(quality_scores) if quality_scores else 0,
                'processable': valid_frames >= self.min_frames_threshold
            }
            
            people_analysis.append(analysis)
        
        # Print analysis
        print(f"\n👥 Multi-person analysis:")
        print(f"   Maximum people detected: {max_people}")
        
        processable_people = sum(1 for p in people_analysis if p['processable'])
        print(f"   Processable people: {processable_people}/{max_people}")
        
        for analysis in people_analysis:
            status = "✅ Processable" if analysis['processable'] else "❌ Skip"
            print(f"   Person {analysis['person_id']}: "
                  f"{analysis['valid_frames']}/{analysis['total_appearances']} valid frames "
                  f"({analysis['valid_rate']*100:.1f}%) - {status}")
        
        return people_analysis
    
    def _process_multiple_people(self, poses_2d, width, height, people_analysis, 
                                enable_filtering, enable_smoothing):
        """Process multiple people sequences"""
        
        all_people_poses_3d = []
        processing_stats = []
        
        processable_people = [p for p in people_analysis if p['processable']]
        total_people = len(processable_people)
        
        print(f"\n🎯 Processing {total_people} people...")
        
        start_time = time.time()
        
        for person_idx, person_analysis in enumerate(people_analysis):
            person_id = person_analysis['person_id']
            
            if not person_analysis['processable']:
                print(f"⏭️  Person {person_id}: Skipped (insufficient data)")
                all_people_poses_3d.append(None)
                processing_stats.append({
                    'person_id': person_id,
                    'status': 'skipped',
                    'reason': 'insufficient_frames',
                    'valid_frames': person_analysis['valid_frames']
                })
                continue
            
            try:
                # Progress tracking
                elapsed_time = time.time() - start_time
                people_per_second = (person_idx + 1) / elapsed_time if elapsed_time > 0 else 0
                remaining_people = total_people - (person_idx + 1)
                eta = remaining_people / people_per_second if people_per_second > 0 else 0
                
                print(f"\r🔄 Processing Person {person_id}... "
                      f"({person_idx + 1}/{total_people} | "
                      f"Speed: {people_per_second:.1f} people/s | "
                      f"ETA: {eta:.1f}s)", end="", flush=True)
                
                # Collect person poses
                person_poses = self._collect_person_poses(poses_2d, person_id)
                
                # Apply filtering if enabled
                if enable_filtering:
                    person_poses = self._filter_poses(person_poses)
                
                # Convert to 3D với progress tracking
                poses_3d = self._convert_to_3d_with_progress(person_poses, width, height, person_id)
                
                # Apply smoothing if enabled
                if enable_smoothing:
                    poses_3d = self._smooth_poses_3d(poses_3d)
                
                all_people_poses_3d.append(poses_3d)
                
                processing_stats.append({
                    'person_id': person_id,
                    'status': 'success',
                    'input_frames': len(person_poses),
                    'output_frames': len(poses_3d),
                    'quality_score': person_analysis['avg_quality']
                })
                
                # Clear the progress line and print success
                print(f"\r✅ Person {person_id}: {len(poses_3d)} frames processed"
                      + " " * 50)  # Clear remaining characters
                
            except Exception as e:
                print(f"\r❌ Person {person_id}: Error - {e}"
                      + " " * 50)  # Clear remaining characters
                all_people_poses_3d.append(None)
                
                processing_stats.append({
                    'person_id': person_id,
                    'status': 'failed',
                    'error': str(e),
                    'valid_frames': person_analysis['valid_frames']
                })
        
        successful_people = sum(1 for stat in processing_stats if stat['status'] == 'success')
        total_time = time.time() - start_time
        print(f"\n✅ Processing completed: {successful_people}/{len(people_analysis)} people successful "
              f"in {total_time:.1f}s ({successful_people/total_time:.1f} people/s)")
        
        return all_people_poses_3d, processing_stats
    
    def _collect_person_poses(self, poses_2d, person_idx):
        """Thu thập poses của một người qua tất cả frames"""
        person_poses = []
        
        for frame_poses in poses_2d:
            if person_idx < len(frame_poses):
                pose = frame_poses[person_idx]
            else:
                # Person không xuất hiện trong frame - tạo dummy pose
                pose = [[0, 0, 0]] * 17  # 17 keypoints với zero values
            
            person_poses.append(pose)
        
        return np.array(person_poses)
    
    def _filter_poses(self, person_poses):
        """Filter low-quality poses"""
        print(f"🔍 Applying pose filtering...")
        
        filtered_poses = []
        original_count = len(person_poses)
        
        for frame_idx, pose in enumerate(person_poses):
            pose_array = np.array(pose)
            
            if pose_array.shape[1] >= 3:  # Has confidence
                valid_keypoints = np.sum(pose_array[:, 2] > self.confidence_threshold)
            else:
                valid_keypoints = np.sum(np.any(pose_array[:, :2] > 0, axis=1))
            
            if valid_keypoints >= self.min_valid_keypoints:
                filtered_poses.append(pose)
            else:
                # Đừng loại bỏ poses - chỉ đánh dấu chúng là cần xử lý đặc biệt
                # Với threshold thấp, hầu hết poses sẽ được giữ lại
                if len(filtered_poses) > 0:
                    filtered_poses.append(filtered_poses[-1])  # Repeat last valid pose
                else:
                    filtered_poses.append(pose)  # Keep original if no previous valid pose
        
        filtered_count = len(filtered_poses)
        print(f"   Kept {filtered_count}/{original_count} frames after filtering")
        
        return np.array(filtered_poses)
    
    def _convert_to_3d(self, person_poses, width, height):
        """Convert 2D poses to 3D using VideoPose3D"""
        print(f"🎯 Converting to 3D...")
        print(f"   Input shape: {person_poses.shape}")
        
        # Extract x,y coordinates (ignore confidence if present)
        if person_poses.shape[-1] > 2:
            poses_2d = person_poses[..., :2]
        else:
            poses_2d = person_poses
        
        # Use VideoPose3D predictor
        poses_3d = self.predictor.predict_3d(poses_2d, width, height)
        
        print(f"   Output shape: {poses_3d.shape}")
        
        return poses_3d
    
    def _convert_to_3d_with_progress(self, person_poses, width, height, person_id):
        """Convert 2D poses to 3D với progress tracking"""
        total_frames = len(person_poses)
        
        print(f"\n🎯 Converting Person {person_id} to 3D ({total_frames} frames)...")
        
        # Extract x,y coordinates (ignore confidence if present)
        if person_poses.shape[-1] > 2:
            poses_2d = person_poses[..., :2]
        else:
            poses_2d = person_poses
        
        # Sử dụng phương thức predict_3d_with_progress nếu có, fallback về phương thức cũ
        if hasattr(self.predictor, 'predict_3d_with_progress'):
            poses_3d = self.predictor.predict_3d_with_progress(poses_2d, width, height)
        else:
            # Fallback: sử dụng phương thức thông thường
            start_time = time.time()
            poses_3d = self.predictor.predict_3d(poses_2d, width, height)
            processing_time = time.time() - start_time
            fps = total_frames / processing_time if processing_time > 0 else 0
            print(f"\r✅ Person {person_id}: {total_frames} frames converted "
                  f"in {processing_time:.1f}s ({fps:.1f} FPS)")
        
        return poses_3d
    
    def _smooth_poses_3d(self, poses_3d, window_size=5):
        """Apply temporal smoothing to 3D poses"""
        print(f"🔄 Applying temporal smoothing (window={window_size})...")
        
        if len(poses_3d) < window_size:
            return poses_3d
        
        smoothed_poses = np.zeros_like(poses_3d)
        
        for i in range(len(poses_3d)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(poses_3d), i + window_size // 2 + 1)
            
            # Average poses in window
            smoothed_poses[i] = np.mean(poses_3d[start_idx:end_idx], axis=0)
        
        print(f"   Applied smoothing to {len(poses_3d)} frames")
        
        return smoothed_poses
    
    def _create_metadata(self, poses_2d_file, poses_2d_data, processing_stats, total_time):
        """Create comprehensive metadata"""
        successful_people = [stat for stat in processing_stats if stat['status'] == 'success']
        
        metadata = {
            'source_file': poses_2d_file,
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': total_time,
                'estimator_settings': {
                    'min_frames_threshold': self.min_frames_threshold,
                    'confidence_threshold': self.confidence_threshold,
                    'min_valid_keypoints': self.min_valid_keypoints
                }
            },
            'video_info': poses_2d_data['video_info'],
            'model_info': self.predictor.get_model_info(),
            'processing_stats': processing_stats,
            'results_summary': {
                'total_people_detected': len(processing_stats),
                'successful_people': len(successful_people),
                'failed_people': len([s for s in processing_stats if s['status'] == 'failed']),
                'skipped_people': len([s for s in processing_stats if s['status'] == 'skipped']),
                'success_rate': len(successful_people) / len(processing_stats) if processing_stats else 0
            },
            'output_format': {
                'coordinate_system': 'human36m',
                'units': 'millimeters',
                'joint_names': self.joint_names,
                'skeleton_connections': self.skeleton_connections,
                'data_structure': 'people -> frames -> joints -> [x, y, z]'
            }
        }
        
        return metadata
    
    def _save_3d_poses(self, all_people_poses_3d, metadata, output_prefix):
        """Save 3D poses và metadata"""
        
        # Filter valid poses
        valid_poses = [poses for poses in all_people_poses_3d if poses is not None]
        
        if not valid_poses:
            print("⚠️ No valid 3D poses to save!")
            return {}
        
        # Prepare output paths
        numpy_path = f"{output_prefix}.npy"
        json_path = f"{output_prefix}.json"
        
        try:
            print(f"💾 Saving 3D poses...")
            start_time = time.time()
            
            # Save as NumPy file
            np.save(numpy_path, valid_poses)
            
            # Prepare JSON data
            json_data = {
                'poses_3d': [poses.tolist() for poses in valid_poses],
                'metadata': metadata
            }
            
            # Save as JSON
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            save_time = time.time() - start_time
            
            output_paths = {
                'numpy': numpy_path,
                'json': json_path
            }
            
            print(f"✅ Saved 3D poses in {save_time:.1f}s:")
            print(f"   NumPy: {numpy_path}")
            print(f"   JSON: {json_path}")
            print(f"   People: {len(valid_poses)}")
            if valid_poses:
                print(f"   Frames per person: {len(valid_poses[0])}")
                print(f"   Joints per frame: {len(valid_poses[0][0])}")
            
            return output_paths
            
        except Exception as e:
            print(f"❌ Error saving 3D poses: {e}")
            raise
    
    def _print_processing_summary(self, metadata, output_paths, total_time):
        """Print comprehensive processing summary"""
        print(f"\n{'='*60}")
        print(f"🎉 3D POSE ESTIMATION COMPLETED")
        print(f"{'='*60}")
        
        # Basic info
        video_info = metadata['video_info']
        print(f"📹 Source: {metadata['source_file']}")
        print(f"📊 Video: {video_info['width']}x{video_info['height']} @ {video_info.get('fps', 'Unknown')} FPS")
        print(f"⏱️  Processing time: {total_time:.2f}s")
        
        # Model info
        model_info = metadata['model_info']
        print(f"🤖 Model: {model_info['architecture']} architecture")
        print(f"   Receptive field: {model_info['receptive_field']} frames")
        print(f"   Parameters: {model_info['total_parameters']:,}")
        
        # Results summary
        summary = metadata['results_summary']
        print(f"\n👥 Results:")
        print(f"   Total people detected: {summary['total_people_detected']}")
        print(f"   Successfully processed: {summary['successful_people']}")
        print(f"   Failed: {summary['failed_people']}")
        print(f"   Skipped: {summary['skipped_people']}")
        print(f"   Success rate: {summary['success_rate']*100:.1f}%")
        
        # Per-person breakdown
        print(f"\n📋 Per-person results:")
        for stat in metadata['processing_stats']:
            person_id = stat['person_id']
            status = stat['status']
            
            if status == 'success':
                frames = stat['output_frames']
                quality = stat['quality_score']
                print(f"   Person {person_id}: ✅ {frames} frames (quality: {quality:.2f})")
            elif status == 'failed':
                error = stat.get('error', 'Unknown error')
                print(f"   Person {person_id}: ❌ Failed - {error}")
            else:  # skipped
                reason = stat.get('reason', 'Unknown reason')
                print(f"   Person {person_id}: ⏭️ Skipped - {reason}")
        
        # Output files
        if output_paths:
            print(f"\n📁 Output files:")
            for format_name, file_path in output_paths.items():
                print(f"   {format_name.title()}: {file_path}")
    
    def process_video_pipeline(self, poses_2d_file, output_prefix="output_3d",
                              enable_filtering=True, enable_smoothing=False):
        """
        Complete pipeline từ 2D poses file đến 3D poses
        Wrapper method for backward compatibility
        
        Args:
            poses_2d_file: File chứa 2D poses
            output_prefix: Prefix cho output files
            enable_filtering: Apply pose filtering
            enable_smoothing: Apply temporal smoothing
            
        Returns:
            tuple: (valid_poses, metadata)
        """
        return self.process_2d_poses_file(
            poses_2d_file, output_prefix, enable_filtering, enable_smoothing
        )
    
    def get_estimator_info(self):
        """Get estimator configuration info"""
        info = {
            'estimator_version': '1.0',
            'videopose3d_available': VIDEOPOSE3D_AVAILABLE,
            'model_info': self.predictor.get_model_info(),
            'settings': {
                'min_frames_threshold': self.min_frames_threshold,
                'confidence_threshold': self.confidence_threshold,
                'min_valid_keypoints': self.min_valid_keypoints
            },
            'joint_names': self.joint_names,
            'skeleton_connections': self.skeleton_connections
        }
        
    def calculate_mpjpe(self, predicted_poses: np.ndarray, ground_truth_poses: np.ndarray) -> float:
        """
        Tính toán Mean Per Joint Position Error (MPJPE)
        
        Args:
            predicted_poses: 3D poses dự đoán (N, 17, 3) hoặc (N, J, 3)
            ground_truth_poses: 3D poses ground truth cùng shape
            
        Returns:
            float: MPJPE value (mm)
        """
        if predicted_poses.shape != ground_truth_poses.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted_poses.shape} vs ground_truth {ground_truth_poses.shape}")
        
        # Tính khoảng cách Euclidean cho mỗi joint trong mỗi frame
        joint_errors = np.linalg.norm(predicted_poses - ground_truth_poses, axis=-1)
        
        # Trung bình trên tất cả joints và frames
        mpjpe = np.mean(joint_errors)
        
        # MPJPE per joint (trung bình trên tất cả frames)
        mpjpe_per_joint = np.mean(joint_errors, axis=0)
        
        print(f"📊 MPJPE calculated: {mpjpe:.3f}mm")
        print(f"📈 MPJPE per joint: {mpjpe_per_joint}")
        
        return mpjpe
    
    def evaluate_with_ground_truth(self, predicted_poses: np.ndarray, 
                                  ground_truth_file: str) -> Dict[str, float]:
        """
        Evaluate predicted poses against ground truth data
        
        Args:
            predicted_poses: Predicted 3D poses
            ground_truth_file: Path to ground truth JSON file
            
        Returns:
            dict: Evaluation metrics including MPJPE
        """
        if not os.path.exists(ground_truth_file):
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")
        
        print(f"📏 Evaluating against ground truth: {ground_truth_file}")
        
        try:
            with open(ground_truth_file, 'r') as f:
                gt_data = json.load(f)
            
            gt_poses = np.array(gt_data['poses_3d'])
            
            if gt_poses.shape != predicted_poses.shape:
                print(f"⚠️ Shape mismatch: predicted {predicted_poses.shape} vs ground truth {gt_poses.shape}")
                # Try to align shapes if possible
                if len(gt_poses.shape) == len(predicted_poses.shape):
                    min_frames = min(gt_poses.shape[0], predicted_poses.shape[0])
                    gt_poses = gt_poses[:min_frames]
                    predicted_poses = predicted_poses[:min_frames]
                    print(f"📐 Aligned to {min_frames} frames")
                else:
                    raise ValueError(f"Cannot align shapes: {predicted_poses.shape} vs {gt_poses.shape}")
            
            # Calculate MPJPE
            mpjpe = self.calculate_mpjpe(predicted_poses, gt_poses)
            
            # Calculate additional metrics
            joint_errors = np.linalg.norm(predicted_poses - gt_poses, axis=-1)
            mpjpe_per_joint = np.mean(joint_errors, axis=0)
            
            # P-MPJPE (Procrustes-aligned MPJPE)
            # This would require more complex alignment, for now we'll use basic MPJPE
            
            evaluation_results = {
                'mpjpe': mpjpe,
                'mpjpe_per_joint': mpjpe_per_joint.tolist(),
                'frames_evaluated': predicted_poses.shape[0],
                'joints_evaluated': predicted_poses.shape[1],
                'evaluation_timestamp': datetime.now().isoformat(),
                'ground_truth_source': ground_truth_file
            }
            
            print(f"✅ Evaluation completed:")
            print(f"   MPJPE: {mpjpe:.3f}mm")
            print(f"   Frames: {evaluation_results['frames_evaluated']}")
            print(f"   Joints: {evaluation_results['joints_evaluated']}")
            
            return evaluation_results
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            raise

def main():
    """Demo sử dụng Pose3DEstimator"""
    print("🎯 3D Pose Estimator Demo")
    print("=" * 50)
    
    # Check VideoPose3D availability
    if not VIDEOPOSE3D_AVAILABLE:
        print("❌ VideoPose3D not available!")
        print("Please setup VideoPose3D repository first:")
        print("1. git clone https://github.com/facebookresearch/VideoPose3D.git")
        print("2. Download pretrained checkpoint")
        print("3. Run this script again")
        return
    
    # Initialize estimator
    try:
        # Try with auto-detected checkpoint
        estimator = Pose3DEstimator()
        
        # Print estimator info
        info = estimator.get_estimator_info()
        print(f"\n📋 Estimator Info:")
        print(f"   VideoPose3D: {info['videopose3d_available']}")
        print(f"   Model architecture: {info['model_info']['architecture']}")
        print(f"   Receptive field: {info['model_info']['receptive_field']} frames")
        
    except Exception as e:
        print(f"❌ Error initializing estimator: {e}")
        return
    
    # File paths
    poses_2d_file = "res/output/output_2d_poses.json"
    
    if not os.path.exists(poses_2d_file):
        print(f"\n❌ 2D poses file not found: {poses_2d_file}")
        return
    
    # Process 2D poses
    try:
        print(f"\n🚀 Processing 2D poses: {poses_2d_file}")
        
        valid_poses, metadata = estimator.process_2d_poses_file(
            poses_2d_file=poses_2d_file,
            output_prefix="res/output/output_3d_poses",
            enable_filtering=True,
            enable_smoothing=True
        )
        
        print(f"\n🎉 Demo completed!")
        print(f"✅ Successfully processed {len(valid_poses)} people")
        
        if valid_poses:
            print(f"📊 Data shape per person: {valid_poses[0].shape}")
            print(f"📁 Output files:")
            print(f"   - res/output/output_3d_poses.json")
            print(f"   - res/output/output_3d_poses.npy")
        
    except Exception as e:
        print(f"❌ Error processing poses: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n💡 Troubleshooting tips:")
        print(f"1. Ensure VideoPose3D repository is properly setup")
        print(f"2. Check pretrained checkpoint is downloaded")
        print(f"3. Verify 2D poses file format")
        print(f"4. Check GPU/CUDA availability")

if __name__ == "__main__":
    main()