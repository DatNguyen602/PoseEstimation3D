# Hướng dẫn sử dụng tính năng đánh giá khiêu vũ với 4 trường mới

## Tổng quan
Hệ thống đã được nâng cấp với khả năng đánh giá chi tiết biểu diễn khiêu vũ dựa trên 4 tiêu chí:
- **Chuẩn nhịp** (Rhythm): Độ nhất quán của tốc độ chuyển động
- **Tư thế** (Posture): Độ thẳng hàng và cân bằng của cơ thể
- **Động tác** (Movement): Độ chính xác của quỹ đạo chuyển động
- **Biểu cảm** (Expression): Độ tự tin và đa dạng của biểu diễn

## Cách sử dụng API

### 1. Gọi API so sánh video với đánh giá chi tiết
```bash
POST /api/compare_videos/
Content-Type: multipart/form-data

# Upload 2 files:
# - user_video: Video của người dùng
# - reference_video: Video tham khảo
# - user_id: (tùy chọn) ID người dùng
# - title: (tùy chọn) Tiêu đề đánh giá
```

### 2. Kết quả trả về
```json
{
  "average_similarity_score": 0.85,
  "dance_scoring_metrics": {
    "rhythm_score": 0.78,
    "posture_score": 0.82,
    "movement_score": 0.88,
    "expression_score": 0.75,
    "total_score": 0.81,
    "baseline_score_percent": 95,
    "sensitivity_score_percent": 88
  },
  "frame_details": [
    {
      "frame": 1,
      "timestamp": 0.033,
      "similarity_score": 0.85,
      "rhythm_score": 0.78,
      "posture_score": 0.82,
      "movement_score": 0.88,
      "expression_score": 0.75,
      "wrong_keypoints_count": 2
    }
  ],
  "excel_download_url": "/res/output/dance_scoring_abc123.xlsx",
  "cloudinary_video_url": "https://res.cloudinary.com/...",
  "total_frames_processed": 150
}
```

## File Excel xuất ra

### Sheet "Tổng quan"
| Tiêu chí | Điểm số |
|----------|----------|
| Điểm tương đồng | 0.85 |
| Chuẩn nhịp | 0.78 |
| Tư thế | 0.82 |
| Động tác | 0.88 |
| Biểu cảm | 0.75 |
| Tổng điểm | 0.81 |

### Sheet "Chi tiết frame"
| Frame | Timestamp | Điểm tương đồng | Chuẩn nhịp | Tư thế | Động tác | Biểu cảm |
|-------|-----------|-----------------|------------|--------|-----------|-----------|
| 1 | 0.033 | 0.85 | 0.78 | 0.82 | 0.88 | 0.75 |
| 2 | 0.067 | 0.87 | 0.80 | 0.84 | 0.89 | 0.77 |
| ... | ... | ... | ... | ... | ... | ... |

### Sheet "Thông số kỹ thuật"
- Baseline Score: Độ chính xác của thuật toán
- Sensitivity Score: Độ nhạy cảm của thuật toán
- Thời gian xử lý
- Thông tin phiên xử lý

## Cách tải file Excel

Khi nhận được kết quả từ API, sử dụng URL trong trường `excel_download_url`:

```javascript
// Ví dụ với JavaScript
const response = await fetch('/api/compare_videos/', {
  method: 'POST',
  body: formData
});

const result = await response.json();
const excelUrl = result.excel_download_url;

// Tải file Excel
window.open(excelUrl, '_blank');
```

## Các chỉ số đánh giá chi tiết

### Chuẩn nhịp (Rhythm Score)
- Đo lường sự nhất quán trong tốc độ chuyển động
- Điểm cao khi các frame có tốc độ tương tự nhau
- Công thức: `1 - (std_deviation / mean_speed)`

### Tư thế (Posture Score)
- Đánh giá độ thẳng hàng của cột sống
- Kiểm tra sự cân bằng của vai và chân
- Các yếu tố: shoulder-hip alignment, shoulder balance, leg alignment

### Động tác (Movement Score)
- So sánh quỹ đạo chuyển động với video tham khảo
- Điểm cao khi keypoints di chuyển đúng hướng
- Sử dụng ngưỡng khoảng cách 0.1 để đánh giá độ chính xác

### Biểu cảm (Expression Score)
- Đo lường độ tự tin dựa trên confidence scores
- Đánh giá sự đa dạng và cân bằng của pose
- Kết hợp confidence + pose variety + balance

## Lưu ý quan trọng

1. **Yêu cầu thư viện**: Cần cài đặt `pandas` và `openpyxl` để xuất Excel
2. **Kích thước file**: File Excel có thể lớn với nhiều frame (>1000 frames)
3. **Thời gian xử lý**: Việc tính toán 4 chỉ số có thể tăng thời gian xử lý ~10-20%
4. **Độ chính xác**: Các chỉ số được tính toán dựa trên dữ liệu pose keypoints từ MediaPipe

## Khắc phục sự cố

### Lỗi thiếu thư viện
```bash
pip install pandas openpyxl
```

### Lỗi tạo Excel
- Kiểm tra dung lượng ổ đĩa
- Đảm bảo thư mục `/res/output/` có quyền ghi
- Kiểm tra phiên bản pandas/openpyxl tương thích

### Kết quả không chính xác
- Kiểm tra chất lượng video đầu vào
- Điều chỉnh ngưỡng threshold trong `pose_comparison.py`
- Kiểm tra độ sáng và góc quay của video
