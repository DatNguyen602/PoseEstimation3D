import requests
import base64
import os

# --- Cấu hình --- 
# !!! QUAN TRỌNG: Hãy thay đổi đường dẫn này thành file ảnh của bạn !!!
USER_IMAGE_PATH = "path/to/your/test_image.jpg" 
API_ENDPOINT = "http://127.0.0.1:8000/api/compare_pose"

# Thông tin video mẫu để so sánh
REFERENCE_VIDEO = "res/input/video.mp4"
REFERENCE_FRAME = 100

def image_to_base64(image_path: str) -> str:
    """Mã hóa một file ảnh thành chuỗi base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file ảnh tại: {image_path}")
        print("Vui lòng cập nhật biến USER_IMAGE_PATH trong script.")
        return None

def run_test():
    """Gửi request đến API và in ra kết quả."""
    print(f"--- Bắt đầu test API so sánh tư thế ---")
    
    # 1. Mã hóa ảnh người dùng
    print(f"Đang mã hóa ảnh: {USER_IMAGE_PATH}")
    user_image_b64 = image_to_base64(USER_IMAGE_PATH)
    if not user_image_b64:
        return

    # 2. Chuẩn bị payload
    payload = {
        "user_image": user_image_b64,
        "reference_video_path": REFERENCE_VIDEO,
        "reference_frame_index": REFERENCE_FRAME
    }
    
    print(f"Gửi request đến: {API_ENDPOINT}")
    print(f"So sánh với frame {REFERENCE_FRAME} của video '{REFERENCE_VIDEO}'")

    # 3. Gửi request POST
    try:
        response = requests.post(API_ENDPOINT, json=payload)
        response.raise_for_status()  # Báo lỗi nếu status code là 4xx hoặc 5xx

        # 4. In kết quả
        print("\n--- Kết quả từ API ---")
        print(f"Status Code: {response.status_code}")
        
        result = response.json()
        print(f"  - Điểm số (Score): {result.get('score'):.2f}")
        print(f"  - Tổng số keypoints: {result.get('total_keypoints')}")
        print(f"  - Các keypoints sai: {result.get('wrong_keypoints')}")
        print("-----------------------\n")

    except requests.exceptions.RequestException as e:
        print(f"\n--- LỖI KẾT NỐI ---")
        print(f"Không thể kết nối đến API. Bạn đã khởi động server chưa?")
        print(f"Lỗi chi tiết: {e}")
        print("-------------------")

if __name__ == "__main__":
    if not os.path.exists(USER_IMAGE_PATH):
        print("Lỗi: Đường dẫn ảnh không tồn tại.")
        print(f"Vui lòng sửa biến 'USER_IMAGE_PATH' trong file test_comparison_api.py")
    else:
        run_test()
