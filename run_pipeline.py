import os
import sys
from pose_detector import PoseDetector
from pose_3d_estimator import Pose3DEstimator

# Thêm thư mục project vào sys.path để import các module từ VideoPose3D
project_root = os.path.dirname(os.path.abspath(__file__))
videopose3d_path = os.path.join(project_root, "VideoPose3D")
if videopose3d_path not in sys.path:
    sys.path.append(videopose3d_path)

def run_full_pipeline(video_path: str, output_dir: str = "res/output", output_basename: str = "output"):
    """
    Chạy toàn bộ quy trình từ video 2D đến ước tính 3D pose.

    Args:
        video_path (str): Đường dẫn đến video đầu vào.
        output_dir (str): Thư mục để lưu các file kết quả.
        output_basename (str): Tên file cơ sở cho các file output, không bao gồm phần mở rộng.

    Returns:
        str: Đường dẫn đến file JSON chứa 3D pose cuối cùng, hoặc None nếu thất bại.
    """
    print(f"🚀 Bắt đầu quy trình cho video: {os.path.basename(video_path)}")

    os.makedirs(output_dir, exist_ok=True)

    # --- Bước 1: Phát hiện 2D Pose ---
    print("--- BƯỚC 1: PHÁT HIỆN 2D POSE ---")
    pose_detector = PoseDetector()
    base_name_2d = os.path.join(output_dir, f"{output_basename}_2d")
    
    all_poses, _, _, _, poses_2d_json_path = pose_detector.detect_poses_from_video(
        video_path=video_path,
        output_path=base_name_2d,
        frame_step=1
    )
    
    if not all_poses:
        print("❌ Không phát hiện được pose nào trong video.")
        return None
    print(f"✅ Đã lưu dữ liệu 2D pose vào: {poses_2d_json_path}")

    # --- Bước 2: Ước tính 3D Pose ---
    print("--- BƯỚC 2: ƯỚC TÍNH 3D POSE ---")
    pose_3d_estimator = Pose3DEstimator()
    base_name_3d = os.path.join(output_dir, f"{output_basename}_3d_poses")

    valid_poses, _ = pose_3d_estimator.process_2d_poses_file(
        poses_2d_file=poses_2d_json_path,
        output_prefix=base_name_3d,
        enable_filtering=True,
        enable_smoothing=True
    )

    final_json_path = base_name_3d + ".json"
    
    # Lấy đường dẫn các file output khác để dọn dẹp
    final_npy_path = base_name_3d + ".npy"
    output_video_path = base_name_2d + "_poses.mp4"


    # Dọn dẹp file 2D JSON trung gian
    if os.path.exists(poses_2d_json_path):
        os.remove(poses_2d_json_path)
        print(f"🗑️ Đã xóa file 2D JSON trung gian: {poses_2d_json_path}")

    if not valid_poses:
        print("❌ Không thể tạo ra 3D pose hợp lệ.")
        # Dọn dẹp các file đã tạo nếu thất bại
        for f in [final_json_path, final_npy_path, output_video_path]:
             if os.path.exists(f):
                os.remove(f)
        return None

    print(f"✅ Đã lưu dữ liệu 3D pose vào: {final_json_path}")
    
    # Trả về đường dẫn các file đã tạo để API có thể dọn dẹp
    generated_files = [final_json_path, final_npy_path, output_video_path]
    return final_json_path, generated_files

if __name__ == '__main__':
    input_video = "res/input/video.mp4"
    output_directory = "res/output"

    if not os.path.exists(input_video):
        print(f"❌ Lỗi: Không tìm thấy video đầu vào tại '{input_video}'")
    else:
        try:
            # Sử dụng tên file video làm basename
            basename = os.path.splitext(os.path.basename(input_video))[0]
            result = run_full_pipeline(input_video, output_directory, output_basename=basename)
            
            if result:
                final_output_path, generated_files = result
                print(f"\n🎉 Quy trình đã hoàn tất thành công!")
                print(f"✅ Kết quả cuối cùng: {final_output_path}")
        except Exception as e:
            print(f"❌ Đã có lỗi xảy ra: {e}")
            import traceback
            traceback.print_exc()