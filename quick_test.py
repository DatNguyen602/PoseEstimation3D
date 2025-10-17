#!/usr/bin/env python3
"""
Script test nhanh để tạo video mẫu và kiểm tra API
"""
import cv2
import numpy as np
import os
import requests

def create_test_video(filename, width=320, height=240, duration=3, fps=15):
    """Tạo video test đơn giản"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(duration * fps):
        # Tạo frame màu
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Thêm màu gradient
        for y in range(height):
            intensity = int(255 * (y / height))
            frame[y, :, 0] = intensity  # Blue
            frame[y, :, 1] = 100        # Green
            frame[y, :, 2] = 150        # Red

        # Thêm text
        cv2.putText(frame, f"Test Frame {i}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"✅ Đã tạo video test: {filename}")

def test_api():
    """Test API endpoints"""
    base_url = "http://localhost:8000"

    print("🔍 Kiểm tra API endpoints...")

    try:
        # Test root endpoint
        response = requests.get(f"{base_url}/")
        print(f"✅ Root: {response.status_code}")

        # Test reference videos list
        response = requests.get(f"{base_url}/api/reference_videos/")
        print(f"✅ Reference videos: {response.status_code}")
        if response.status_code == 200:
            videos = response.json()
            print(f"   Số video tham khảo: {len(videos)}")

    except Exception as e:
        print(f"❌ Lỗi kết nối API: {e}")
        return False

    return True

def main():
    """Main test function"""
    print("🚀 Bắt đầu test setup...")

    # Tạo thư mục cần thiết
    os.makedirs("reference_videos", exist_ok=True)
    os.makedirs("res/output", exist_ok=True)

    # Tạo video test
    test_video = "reference_videos/test_video.mp4"
    if not os.path.exists(test_video):
        print("🎬 Tạo video test...")
        create_test_video(test_video)

    # Test API
    if test_api():
        print("✅ Setup hoàn tất!")
        print("🌐 Mở trình duyệt và test:")
        print("   http://localhost:8000/simple_test.html")
        print("   Hoặc mở file HTML bất kỳ để test giao diện")
    else:
        print("❌ Cần khởi động server trước: python3 main_api.py")

if __name__ == "__main__":
    main()
