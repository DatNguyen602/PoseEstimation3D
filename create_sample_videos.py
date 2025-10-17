#!/usr/bin/env python3
"""
Script tạo video mẫu để test API
"""
import cv2
import numpy as np
import os

def create_sample_video(filename, duration=5, fps=30):
    """Tạo video mẫu với các hình tròn màu để test pose detection"""

    # Thông số video
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Tạo video writer
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(duration * fps):
        # Tạo frame với background màu
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Thêm gradient background
        for y in range(height):
            color = int(255 * (y / height))
            frame[y, :, 0] = color  # Blue channel
            frame[y, :, 1] = 100    # Green channel
            frame[y, :, 2] = 150    # Red channel

        # Vẽ các hình tròn (giả lập pose keypoints)
        center_x, center_y = width // 2, height // 2

        # Frame hiện tại trong cycle
        frame_in_cycle = i % fps

        if frame_in_cycle < fps // 2:
            # Nửa đầu: pose đứng thẳng
            positions = [
                (center_x, center_y - 100),  # Head
                (center_x, center_y - 50),   # Neck
                (center_x, center_y),        # Body center
                (center_x - 50, center_y + 50),  # Left hand
                (center_x + 50, center_y + 50),  # Right hand
                (center_x - 30, center_y + 100), # Left foot
                (center_x + 30, center_y + 100), # Right foot
            ]
        else:
            # Nửa sau: pose squat xuống
            positions = [
                (center_x, center_y - 80),   # Head (lower)
                (center_x, center_y - 30),   # Neck (lower)
                (center_x, center_y + 20),   # Body center (lower)
                (center_x - 60, center_y + 40),  # Left hand (wider)
                (center_x + 60, center_y + 40),  # Right hand (wider)
                (center_x - 40, center_y + 80),  # Left foot (wider)
                (center_x + 40, center_y + 80),  # Right foot (wider)
            ]

        # Vẽ các keypoints
        for j, (x, y) in enumerate(positions):
            cv2.circle(frame, (x, y), 15, (0, 255, 0), -1)
            cv2.putText(frame, str(j), (x - 10, y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Vẽ connections giữa keypoints (skeleton)
        connections = [
            (0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (2, 6)
        ]

        for start_idx, end_idx in connections:
            if start_idx < len(positions) and end_idx < len(positions):
                start = positions[start_idx]
                end = positions[end_idx]
                cv2.line(frame, start, end, (255, 0, 0), 3)

        # Thêm text hiển thị frame number
        cv2.putText(frame, f"Frame: {i}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Ghi frame vào video
        out.write(frame)

    # Giải phóng video writer
    out.release()
    print(f"✅ Đã tạo video mẫu: {filename}")

def create_sample_videos():
    """Tạo các video mẫu để test"""

    # Tạo thư mục
    os.makedirs("reference_videos", exist_ok=True)
    os.makedirs("res/output", exist_ok=True)

    print("🎬 Đang tạo video mẫu...")

    # Video tham khảo (5 giây)
    create_sample_video("reference_videos/sample_reference.mp4", duration=3, fps=15)

    # Video người dùng (3 giây)
    create_sample_video("reference_videos/sample_user.mp4", duration=2, fps=15)

    print("✅ Hoàn thành tạo video mẫu!")
    print("📁 Video tham khảo: reference_videos/sample_reference.mp4")
    print("📁 Video người dùng: reference_videos/sample_user.mp4")

if __name__ == "__main__":
    create_sample_videos()
