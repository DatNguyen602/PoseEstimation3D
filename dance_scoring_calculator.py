import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple
import math

logger = logging.getLogger(__name__)

class DanceScoringCalculator:
    """Tính toán các chỉ số đánh giá khiêu vũ chi tiết"""

    def __init__(self):
        self.keypoint_indices = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16, 'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26, 'left_ankle': 27, 'right_ankle': 28
        }

    def calculate_rhythm_score(self, pose_data: List[Dict], timestamps: List[float]) -> float:
        """
        Tính điểm chuẩn nhịp dựa trên sự nhất quán của chuyển động

        Args:
            pose_data: Danh sách dữ liệu pose của từng frame
            timestamps: Danh sách thời gian của từng frame

        Returns:
            Điểm chuẩn nhịp (0-100)
        """
        if len(pose_data) < 3 or len(timestamps) < 3:
            return 0.0

        try:
            # Tính tốc độ trung bình của các keypoints chính
            movement_speeds = []

            for i in range(1, len(pose_data)):
                if i < len(timestamps):
                    time_diff = timestamps[i] - timestamps[i-1]
                    if time_diff > 0:
                        speed = self._calculate_movement_speed(pose_data[i-1], pose_data[i])
                        movement_speeds.append(speed / time_diff)  # Normalize theo thời gian

            if not movement_speeds:
                return 0.0

            # Tính độ ổn định của tốc độ
            avg_speed = np.mean(movement_speeds)
            if avg_speed == 0:
                return 50.0  # Điểm trung bình nếu không có chuyển động

            # Tính độ lệch chuẩn của tốc độ (thấp hơn = ổn định hơn)
            speed_std = np.std(movement_speeds)
            consistency = max(0, 1.0 - (speed_std / avg_speed))

            # Áp dụng hàm sigmoid để chuyển đổi sang thang điểm 0-100
            rhythm_score = self._sigmoid_scale(consistency, 0.5, 0.2) * 100

            logger.debug(f"Rhythm score calculated: {rhythm_score:.2f} (consistency: {consistency:.3f})")
            return rhythm_score

        except Exception as e:
            logger.error(f"Error calculating rhythm score: {e}")
            return 0.0

    def calculate_posture_score(self, keypoints: np.ndarray) -> float:
        """
        Tính điểm tư thế dựa trên góc độ và vị trí các bộ phận

        Args:
            keypoints: Mảng numpy chứa tọa độ các keypoints

        Returns:
            Điểm tư thế (0-100)
        """
        if keypoints is None or len(keypoints) < 66:  # 33 keypoints * 2 coordinates
            return 0.0

        try:
            posture_indicators = []

            # 1. Kiểm tra độ thẳng của cột sống (shoulder - hip alignment)
            left_shoulder = keypoints[11*2:11*2+2]
            right_shoulder = keypoints[12*2:12*2+2]
            left_hip = keypoints[23*2:23*2+2]
            right_hip = keypoints[24*2:24*2+2]

            if self._check_valid_keypoints([left_shoulder, right_shoulder, left_hip, right_hip]):
                alignment_score = self._calculate_spine_alignment(left_shoulder, right_shoulder, left_hip, right_hip)
                posture_indicators.append(alignment_score)

            # 2. Kiểm tra độ cân bằng của vai
            if self._check_valid_keypoints([left_shoulder, right_shoulder]):
                shoulder_balance = self._calculate_shoulder_balance(left_shoulder, right_shoulder)
                posture_indicators.append(shoulder_balance)

            # 3. Kiểm tra độ thẳng của chân
            left_knee = keypoints[25*2:25*2+2]
            right_knee = keypoints[26*2:26*2+2]
            left_ankle = keypoints[27*2:27*2+2]
            right_ankle = keypoints[28*2:28*2+2]

            if self._check_valid_keypoints([left_knee, right_knee, left_ankle, right_ankle]):
                leg_alignment = self._calculate_leg_alignment(left_knee, right_knee, left_ankle, right_ankle)
                posture_indicators.append(leg_alignment)

            # Tính điểm trung bình
            if posture_indicators:
                avg_score = np.mean(posture_indicators)
                posture_score = self._sigmoid_scale(avg_score, 0.7, 0.15) * 100
                logger.debug(f"Posture score calculated: {posture_score:.2f}")
                return posture_score

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating posture score: {e}")
            return 0.0

    def calculate_movement_score(self, user_keypoints: np.ndarray, ref_keypoints: np.ndarray) -> float:
        """
        Tính điểm động tác dựa trên độ chính xác của quỹ đạo chuyển động

        Args:
            user_keypoints: Keypoints của người dùng
            ref_keypoints: Keypoints tham khảo

        Returns:
            Điểm động tác (0-100)
        """
        if user_keypoints is None or ref_keypoints is None:
            return 0.0

        try:
            # Các keypoints quan trọng để đánh giá động tác
            keypoint_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # Vai, khuỷu tay, cổ tay, hông, đầu gối, cổ chân

            movement_scores = []

            for idx in keypoint_indices:
                user_kp = user_keypoints[idx*2:idx*2+2]
                ref_kp = ref_keypoints[idx*2:idx*2+2]

                if self._check_valid_keypoints([user_kp, ref_kp]):
                    distance = np.linalg.norm(user_kp - ref_kp)
                    # Normalize theo threshold 0.1 (khoảng cách tối đa chấp nhận được)
                    accuracy = max(0, 1.0 - distance / 0.1)
                    movement_scores.append(accuracy)

            if movement_scores:
                avg_score = np.mean(movement_scores)
                movement_score = self._sigmoid_scale(avg_score, 0.8, 0.1) * 100
                logger.debug(f"Movement score calculated: {movement_score:.2f}")
                return movement_score

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating movement score: {e}")
            return 0.0

    def calculate_expression_score(self, keypoints: np.ndarray, pose_results) -> float:
        """
        Tính điểm biểu cảm dựa trên độ tự tin và sự biểu đạt của pose

        Args:
            keypoints: Tọa độ các keypoints
            pose_results: Kết quả từ MediaPipe pose detection

        Returns:
            Điểm biểu cảm (0-100)
        """
        if pose_results is None or not pose_results.pose_landmarks:
            return 0.0

        try:
            # 1. Tính độ tự tin dựa trên visibility scores
            confidence_scores = []
            for landmark in pose_results.pose_landmarks.landmark[:15]:  # 15 keypoints đầu tiên
                if landmark.visibility > 0:
                    confidence_scores.append(landmark.visibility)

            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0

            # 2. Tính độ đa dạng của pose (độ mở rộng của các bộ phận)
            pose_variety = self._calculate_pose_variety(keypoints)

            # 3. Kiểm tra độ cân bằng tổng thể
            balance_score = self._calculate_balance_score(keypoints)

            # Kết hợp các yếu tố
            expression_score = (avg_confidence * 0.5 + pose_variety * 0.3 + balance_score * 0.2) * 100

            logger.debug(f"Expression score calculated: {expression_score:.2f}")
            return expression_score

        except Exception as e:
            logger.error(f"Error calculating expression score: {e}")
            return 0.0

    def _calculate_movement_speed(self, pose1: Dict, pose2: Dict) -> float:
        """Tính tốc độ chuyển động giữa 2 frame"""
        if not pose1 or not pose2:
            return 0.0

        # Tính khoảng cách trung bình của các keypoints chính
        keypoint_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        total_distance = 0
        valid_points = 0

        for idx in keypoint_indices:
            kp1 = pose1.get('keypoints', [])[idx*3:idx*3+3] if 'keypoints' in pose1 else []
            kp2 = pose2.get('keypoints', [])[idx*3:idx*3+3] if 'keypoints' in pose2 else []

            if len(kp1) >= 2 and len(kp2) >= 2:
                distance = math.sqrt((kp1[0] - kp2[0])**2 + (kp1[1] - kp2[1])**2)
                total_distance += distance
                valid_points += 1

        return total_distance / valid_points if valid_points > 0 else 0.0

    def _check_valid_keypoints(self, keypoints_list: List) -> bool:
        """Kiểm tra danh sách keypoints có hợp lệ không"""
        for kp in keypoints_list:
            if not isinstance(kp, np.ndarray) or len(kp) != 2:
                return False
        return True

    def _calculate_spine_alignment(self, left_shoulder, right_shoulder, left_hip, right_hip):
        """Tính độ thẳng của cột sống"""
        # Tính trung điểm vai và hông
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2

        # Tính độ lệch theo trục x (vai và hông nên thẳng hàng theo chiều dọc)
        x_alignment = 1.0 - abs(shoulder_center[0] - hip_center[0])

        # Tính khoảng cách giữa vai và hông (độ dài cột sống hợp lý)
        spine_length = np.linalg.norm(shoulder_center - hip_center)
        optimal_length = 0.3  # Giả sử chiều cao chuẩn hóa
        length_score = 1.0 - abs(spine_length - optimal_length) / optimal_length

        return (x_alignment * 0.7 + length_score * 0.3)

    def _calculate_shoulder_balance(self, left_shoulder, right_shoulder):
        """Tính độ cân bằng của vai"""
        # Vai nên cân bằng (cùng chiều cao)
        height_diff = abs(left_shoulder[1] - right_shoulder[1])
        return max(0, 1.0 - height_diff * 5)  # Nhân 5 để tăng sensitivity

    def _calculate_leg_alignment(self, left_knee, right_knee, left_ankle, right_ankle):
        """Tính độ thẳng của chân"""
        # Tính độ thẳng hàng đầu gối - cổ chân
        left_leg_alignment = self._calculate_limb_alignment(left_knee, left_ankle)
        right_leg_alignment = self._calculate_limb_alignment(right_knee, right_ankle)

        return (left_leg_alignment + right_leg_alignment) / 2

    def _calculate_limb_alignment(self, joint1, joint2):
        """Tính độ thẳng hàng của 2 khớp"""
        # Vector giữa 2 khớp
        vector = joint2 - joint1
        # Độ thẳng hàng lý tưởng là theo trục y (chiều dọc)
        vertical_alignment = abs(vector[1]) / np.linalg.norm(vector)
        return vertical_alignment

    def _calculate_pose_variety(self, keypoints: np.ndarray) -> float:
        """Tính độ đa dạng của pose dựa trên sự phân bố các keypoints"""
        if keypoints is None or len(keypoints) < 10:
            return 0.0

        # Tính bounding box của pose
        x_coords = keypoints[::2]  # Chỉ lấy tọa độ x
        y_coords = keypoints[1::2]  # Chỉ lấy tọa độ y

        if len(x_coords) == 0 or len(y_coords) == 0:
            return 0.0

        # Tính độ rộng của pose
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)

        # Tính tỷ lệ khung hình (aspect ratio)
        aspect_ratio = width / height if height > 0 else 0

        # Điểm cao hơn cho pose có tỷ lệ cân đối
        optimal_ratio = 0.6  # Tỷ lệ vàng cho vũ đạo
        variety_score = 1.0 - abs(aspect_ratio - optimal_ratio) / optimal_ratio

        return max(0, variety_score)

    def _calculate_balance_score(self, keypoints: np.ndarray) -> float:
        """Tính độ cân bằng tổng thể của pose"""
        if keypoints is None or len(keypoints) < 20:
            return 0.0

        # Tính trọng tâm của pose
        left_side = keypoints[::2][:len(keypoints)//4]  # 1/4 keypoints bên trái
        right_side = keypoints[::2][len(keypoints)//2:len(keypoints)//2 + len(keypoints)//4]  # 1/4 keypoints bên phải

        if len(left_side) > 0 and len(right_side) > 0:
            left_center = np.mean(left_side)
            right_center = np.mean(right_side)
            balance = 1.0 - abs(left_center - right_center)
            return balance

        return 0.5

    def _sigmoid_scale(self, value: float, midpoint: float = 0.5, steepness: float = 0.1) -> float:
        """Áp dụng hàm sigmoid để scale giá trị về khoảng 0-1"""
        return 1 / (1 + np.exp(-(value - midpoint) / steepness))

def export_dance_scoring_to_excel(result_data: Dict[str, Any], request_id: str, output_dir: str) -> str:
    """
    Xuất kết quả đánh giá khiêu vũ ra file Excel

    Args:
        result_data: Dữ liệu kết quả từ API
        request_id: ID của request
        output_dir: Thư mục đầu ra

    Returns:
        Đường dẫn file Excel
    """
    try:
        excel_filename = f"dance_scoring_{request_id}.xlsx"
        excel_path = os.path.join(output_dir, excel_filename)

        calculator = DanceScoringCalculator()

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet Tổng quan
            overview_data = {
                'Thời gian xử lý (giây)': result_data.get('processing_time', 0),
                'Điểm tương đồng trung bình': result_data.get('average_similarity_score', 0),
                'Điểm chuẩn nhịp': result_data.get('rhythm_score', 0),
                'Điểm tư thế': result_data.get('posture_score', 0),
                'Điểm động tác': result_data.get('movement_score', 0),
                'Điểm biểu cảm': result_data.get('expression_score', 0),
                'Tổng điểm': result_data.get('total_score', 0),
                'Số frame xử lý': result_data.get('total_frames_processed', 0),
                'Baseline Score (%)': result_data.get('dance_scoring_metrics', {}).get('baseline_score_percent', 0),
                'Sensitivity Score (%)': result_data.get('dance_scoring_metrics', {}).get('sensitivity_score_percent', 0)
            }

            overview_df = pd.DataFrame([overview_data])
            overview_df.to_excel(writer, sheet_name='Tổng quan', index=False)

            # Sheet Chi tiết từng frame (nếu có dữ liệu)
            if 'frame_details' in result_data and result_data['frame_details']:
                details_df = pd.DataFrame(result_data['frame_details'])
                details_df.to_excel(writer, sheet_name='Chi tiết frame', index=False)

            # Sheet Thông số kỹ thuật
            technical_data = result_data.get('dance_scoring_metrics', {})
            if technical_data:
                technical_df = pd.DataFrame([technical_data])
                technical_df.to_excel(writer, sheet_name='Thông số kỹ thuật', index=False)

        logger.info(f"✅ Dance scoring Excel exported successfully: {excel_path}")
        return excel_path

    except Exception as e:
        logger.error(f"❌ Error exporting dance scoring to Excel: {e}")
        raise
