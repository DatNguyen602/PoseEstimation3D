import time
import logging
import os
from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ProcessingMetrics:
    """Lớp chứa tất cả các metrics hiệu suất xử lý"""

    # Thông tin cơ bản
    video_path: str = ""
    video_duration: float = 0.0
    video_fps: float = 0.0
    total_frames: int = 0
    processed_frames: int = 0

    # Thời gian xử lý (giây)
    total_processing_time: float = 0.0
    pose_detection_time: float = 0.0
    pose_3d_estimation_time: float = 0.0
    io_operations_time: float = 0.0

    # Tốc độ xử lý (FPS)
    pose_detection_fps: float = 0.0
    pose_3d_estimation_fps: float = 0.0
    overall_fps: float = 0.0

    # Frame processing details
    frames_with_poses: int = 0
    total_people_detected: int = 0
    avg_people_per_frame: float = 0.0

    # 3D pose quality metrics
    valid_3d_poses: int = 0
    total_3d_joints: int = 0
    avg_3d_confidence: float = 0.0

    # MPJPE (Mean Per Joint Position Error) - nếu có ground truth
    mpjpe_available: bool = False
    mpjpe_value: float = 0.0
    mpjpe_per_joint: List[float] = field(default_factory=list)

    # Thời gian bắt đầu và kết thúc
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def calculate_derived_metrics(self):
        """Tính toán các metrics phụ thuộc"""
        if self.total_processing_time > 0:
            self.overall_fps = self.processed_frames / self.total_processing_time

        if self.pose_detection_time > 0:
            self.pose_detection_fps = self.processed_frames / self.pose_detection_time

        if self.pose_3d_estimation_time > 0:
            self.pose_3d_estimation_fps = self.valid_3d_poses / self.pose_3d_estimation_time

        if self.processed_frames > 0:
            self.avg_people_per_frame = self.total_people_detected / self.processed_frames

        if self.total_3d_joints > 0:
            self.avg_3d_confidence = self.total_3d_joints / self.valid_3d_poses if self.valid_3d_poses > 0 else 0.0

    def get_processing_time_for_video_length(self, target_duration_seconds: float = 60.0) -> float:
        """Tính thời gian xử lý dự kiến cho video dài target_duration_seconds"""
        if self.video_fps > 0 and self.overall_fps > 0:
            frames_in_target = target_duration_seconds * self.video_fps
            return frames_in_target / self.overall_fps
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành dictionary để serialize"""
        self.calculate_derived_metrics()

        # Calculate rates safely to avoid division by zero
        detection_rate = 0.0
        if self.processed_frames > 0:
            detection_rate = (self.frames_with_poses / self.processed_frames) * 100

        pose_success_rate = 0.0
        if self.processed_frames > 0:
            pose_success_rate = (self.valid_3d_poses / self.processed_frames) * 100

        processing_efficiency = 0.0
        if self.total_frames > 0:
            processing_efficiency = (self.processed_frames / self.total_frames) * 100

        return {
            "video_info": {
                "video_path": self.video_path,
                "video_duration_seconds": self.video_duration,
                "video_fps": self.video_fps,
                "total_frames": self.total_frames,
                "processed_frames": self.processed_frames,
                "processing_efficiency": f"{processing_efficiency:.1f}%"
            },
            "processing_times": {
                "total_processing_time_seconds": round(self.total_processing_time, 3),
                "pose_detection_time_seconds": round(self.pose_detection_time, 3),
                "pose_3d_estimation_time_seconds": round(self.pose_3d_estimation_time, 3),
                "io_operations_time_seconds": round(self.io_operations_time, 3),
                "estimated_time_for_1min_video": round(self.get_processing_time_for_video_length(60.0), 3)
            },
            "processing_speeds": {
                "overall_fps": round(self.overall_fps, 2),
                "pose_detection_fps": round(self.pose_detection_fps, 2),
                "pose_3d_estimation_fps": round(self.pose_3d_estimation_fps, 2)
            },
            "detection_statistics": {
                "frames_with_poses": self.frames_with_poses,
                "total_people_detected": self.total_people_detected,
                "avg_people_per_frame": round(self.avg_people_per_frame, 2),
                "detection_rate": f"{detection_rate:.1f}%"
            },
            "pose_3d_quality": {
                "valid_3d_poses": self.valid_3d_poses,
                "total_3d_joints": self.total_3d_joints,
                "avg_3d_confidence": round(self.avg_3d_confidence, 3),
                "pose_success_rate": f"{pose_success_rate:.1f}%"
            },
            "accuracy_metrics": {
                "mpjpe_available": self.mpjpe_available,
                "mpjpe_value": round(self.mpjpe_value, 3) if self.mpjpe_available else None,
                "mpjpe_per_joint": [round(x, 3) for x in self.mpjpe_per_joint] if self.mpjpe_available else None
            },
            "timing_info": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "total_duration_seconds": round((self.end_time - self.start_time).total_seconds(), 3) if self.end_time else None
            }
        }


class PerformanceMetricsCollector:
    """Lớp thu thập và quản lý các metrics hiệu suất"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = ProcessingMetrics()
        self._start_times: Dict[str, float] = {}
        self._frame_timings: List[float] = []
        self._pose_counts: List[int] = []

    def start_processing(self, video_path: str, video_info: Dict = None):
        """Bắt đầu thu thập metrics cho quá trình xử lý"""
        self.metrics = ProcessingMetrics(video_path=video_path)
        self.metrics.start_time = datetime.now()

        if video_info:
            self.metrics.video_duration = video_info.get('duration', 0.0)
            self.metrics.video_fps = video_info.get('fps', 0.0)
            self.metrics.total_frames = video_info.get('total_frames', 0)

        self.logger.info(f"🚀 Started processing video: {os.path.basename(video_path)}")
        self.logger.info(f"📊 Video info: {self.metrics.video_duration:.2f}s, {self.metrics.video_fps:.2f} FPS, {self.metrics.total_frames} frames")

    def start_phase(self, phase_name: str):
        """Bắt đầu đo thời gian cho một phase cụ thể"""
        self._start_times[phase_name] = time.time()

    def end_phase(self, phase_name: str):
        """Kết thúc đo thời gian cho một phase cụ thể"""
        if phase_name in self._start_times:
            duration = time.time() - self._start_times[phase_name]

            if phase_name == "pose_detection":
                self.metrics.pose_detection_time += duration
            elif phase_name == "pose_3d_estimation":
                self.metrics.pose_3d_estimation_time += duration
            elif phase_name == "io_operations":
                self.metrics.io_operations_time += duration

            self.logger.info(f"⏱️ {phase_name} completed in {duration:.3f}s")

    def record_frame_processing(self, frame_idx: int, processing_time: float, pose_count: int = 0):
        """Ghi nhận thông tin xử lý của từng frame"""
        self._frame_timings.append(processing_time)
        self._pose_counts.append(pose_count)

        # Tăng số frame đã xử lý
        self.metrics.processed_frames += 1

        if pose_count > 0:
            self.metrics.frames_with_poses += 1
        self.metrics.total_people_detected += pose_count

    def record_3d_pose_result(self, pose_3d_data: Any):
        """Ghi nhận kết quả 3D pose - chỉ tính poses thực sự hợp lệ"""
        # Kiểm tra pose có thực sự hợp lệ không (không phải toàn số 0)
        if pose_3d_data is not None:
            # Chuyển thành numpy array để kiểm tra
            if not hasattr(pose_3d_data, 'shape'):
                pose_array = np.array(pose_3d_data)
            else:
                pose_array = pose_3d_data
            
            # Kiểm tra có ít nhất 1 joint không phải số 0
            if np.any(pose_array != 0):
                self.metrics.valid_3d_poses += 1
                
                if len(pose_array.shape) >= 2:
                    joints_count = pose_array.shape[-2] if len(pose_array.shape) > 2 else pose_array.shape[0]
                    self.metrics.total_3d_joints += joints_count
                
                print(f"✅ Valid 3D pose recorded (joints: {joints_count})")
            else:
                print(f"⚠️ Invalid 3D pose (all zeros) - skipped")
        else:
            print(f"⚠️ None 3D pose - skipped")

    def calculate_mpjpe(self, predicted_poses: np.ndarray, ground_truth_poses: np.ndarray) -> float:
        """Tính toán MPJPE (Mean Per Joint Position Error)"""
        if predicted_poses.shape != ground_truth_poses.shape:
            self.logger.warning(f"❌ Shape mismatch for MPJPE calculation: {predicted_poses.shape} vs {ground_truth_poses.shape}")
            return 0.0

        # Tính khoảng cách Euclidean cho mỗi joint
        joint_errors = np.linalg.norm(predicted_poses - ground_truth_poses, axis=-1)

        if len(joint_errors.shape) > 1:
            # Trung bình trên tất cả các joint và frame
            mpjpe = np.mean(joint_errors)
            # MPJPE per joint (trung bình trên tất cả frame)
            mpjpe_per_joint = np.mean(joint_errors, axis=0).tolist()
        else:
            mpjpe = joint_errors
            mpjpe_per_joint = [joint_errors]

        self.metrics.mpjpe_available = True
        self.metrics.mpjpe_value = float(mpjpe)
        self.metrics.mpjpe_per_joint = mpjpe_per_joint

        self.logger.info(f"📊 MPJPE calculated: {mpjpe:.3f} (available for {len(mpjpe_per_joint)} joints)")

        return mpjpe

    def finish_processing(self):
        """Hoàn thành thu thập metrics và tính toán các giá trị cuối cùng"""
        self.metrics.end_time = datetime.now()
        self.metrics.total_processing_time = sum(self._frame_timings) if self._frame_timings else 0.0

        self.metrics.calculate_derived_metrics()

        total_time = (self.metrics.end_time - self.metrics.start_time).total_seconds()

        detection_rate = 0.0
        if self.metrics.processed_frames > 0:
            detection_rate = (self.metrics.frames_with_poses / self.metrics.processed_frames) * 100

        self.logger.info(f"✅ Processing completed in {total_time:.3f}s")
        self.logger.info(f"🚀 Overall FPS: {self.metrics.overall_fps:.2f}")
        self.logger.info(f"📈 Detection rate: {self.metrics.frames_with_poses}/{self.metrics.processed_frames} frames ({detection_rate:.1f}%)")
        self.logger.info(f"👥 Average people per frame: {self.metrics.avg_people_per_frame:.2f}")

        if self.metrics.mpjpe_available:
            self.logger.info(f"🎯 3D Accuracy (MPJPE): {self.metrics.mpjpe_value:.3f}")

    def get_summary_report(self) -> str:
        """Tạo báo cáo tóm tắt về hiệu suất"""
        self.metrics.calculate_derived_metrics()

        detection_rate = 0.0
        if self.metrics.processed_frames > 0:
            detection_rate = (self.metrics.frames_with_poses / self.metrics.processed_frames) * 100

        pose_success_rate = 0.0
        if self.metrics.processed_frames > 0:
            pose_success_rate = (self.metrics.valid_3d_poses / self.metrics.processed_frames) * 100

        lines = [
            "\n" + "="*60,
            "📊 PERFORMANCE METRICS REPORT",
            "="*60,
            f"🎬 Video: {os.path.basename(self.metrics.video_path)}",
            f"⏱️  Processing Time: {self.metrics.total_processing_time:.2f}s",
            f"🚀 Overall FPS: {self.metrics.overall_fps:.2f}",
            f"📈 Detection Rate: {self.metrics.frames_with_poses}/{self.metrics.processed_frames} ({detection_rate:.1f}%)",
            f"👥 Avg People/Frame: {self.metrics.avg_people_per_frame:.2f}",
            f"🎯 3D Pose Success Rate: {self.metrics.valid_3d_poses}/{self.metrics.processed_frames} ({pose_success_rate:.1f}%)"
        ]

        if self.metrics.mpjpe_available:
            lines.extend([
                f"📏 MPJPE (3D Accuracy): {self.metrics.mpjpe_value:.3f}mm"
            ])

        lines.append("="*60)

        return "\n".join(lines)

    def save_metrics_to_file(self, output_path: str):
        """Lưu metrics vào file JSON"""
        try:
            import json
            metrics_dict = self.metrics.to_dict()

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"💾 Performance metrics saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"❌ Error saving metrics to file: {e}")


# Global metrics collector instance
metrics_collector = PerformanceMetricsCollector()
