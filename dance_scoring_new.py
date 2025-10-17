import cv2
import numpy as np
import logging
import time
import json
from typing import Dict, List, Any
from pose_comparison import PoseComparison

class DanceScoringSystem:
    """Hệ thống chấm điểm điệu múa với logging chi tiết"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_baseline_score(self, reference_video_path: str) -> Dict[str, Any]:
        """Tính điểm số tham chiếu (Baseline Score) - so sánh video với chính nó"""
        start_time = time.time()
        self.logger.info(f"🔄 Bắt đầu tính baseline score cho: {reference_video_path}")

        try:
            ref_cap = cv2.VideoCapture(reference_video_path)
            if not ref_cap.isOpened():
                raise ValueError(f"Không thể mở video: {reference_video_path}")

            total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            comparison = PoseComparison(reference_video_path)

            baseline_scores = []
            for frame_idx in range(min(total_frames, 100)):  # Giới hạn 100 frames để tránh timeout
                ret, frame = ref_cap.read()
                if not ret:
                    break

                current_keypoints, _ = comparison._extract_keypoints(frame)
                ref_keypoints, _ = comparison._extract_keypoints(frame)

                if current_keypoints is not None and ref_keypoints is not None:
                    score, _ = comparison._calculate_score(current_keypoints, ref_keypoints)
                    baseline_scores.append(score)

            ref_cap.release()

            avg_score = np.mean(baseline_scores) if baseline_scores else 0.0
            total_time = time.time() - start_time

            return {
                'baseline_score_percent': round(avg_score * 100, 2),
                'processing_time': round(total_time, 3),
                'frames_processed': len(baseline_scores)
            }

        except Exception as e:
            self.logger.error(f"❌ Lỗi baseline scoring: {str(e)}")
            return {'error': str(e), 'baseline_score_percent': 0.0}

    def calculate_sensitivity_score(self, reference_video_path: str, incorrect_video_path: str) -> Dict[str, Any]:
        """Tính độ nhạy thuật toán - so sánh video gốc với video sai"""
        start_time = time.time()
        self.logger.info(f"🔄 Bắt đầu tính sensitivity score")

        try:
            ref_cap = cv2.VideoCapture(reference_video_path)
            incorrect_cap = cv2.VideoCapture(incorrect_video_path)

            total_frames = min(
                int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(incorrect_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            )

            comparison = PoseComparison(reference_video_path)
            sensitivity_scores = []

            for frame_idx in range(min(total_frames, 100)):  # Giới hạn 100 frames
                ret_ref, ref_frame = ref_cap.read()
                ret_incorrect, incorrect_frame = incorrect_cap.read()

                if not ret_ref or not ret_incorrect:
                    break

                ref_keypoints, _ = comparison._extract_keypoints(ref_frame)
                incorrect_keypoints, _ = comparison._extract_keypoints(incorrect_frame)

                if ref_keypoints is not None and incorrect_keypoints is not None:
                    score, _ = comparison._calculate_score(incorrect_keypoints, ref_keypoints)
                    sensitivity_scores.append(score)

            ref_cap.release()
            incorrect_cap.release()

            avg_score = np.mean(sensitivity_scores) if sensitivity_scores else 0.0
            total_time = time.time() - start_time

            return {
                'sensitivity_score_percent': round(avg_score * 100, 2),
                'processing_time': round(total_time, 3),
                'frames_processed': len(sensitivity_scores)
            }

        except Exception as e:
            self.logger.error(f"❌ Lỗi sensitivity scoring: {str(e)}")
            return {'error': str(e), 'sensitivity_score_percent': 0.0}

# Global instance
dance_scorer = DanceScoringSystem()
