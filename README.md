# 🎯 Pose Estimation & Video Comparison API

Đây là một hệ thống API mạnh mẽ để phân tích và so sánh pose (tư thế) từ video, được xây dựng bằng FastAPI với khả năng xử lý real-time và phân tích độ chính xác.

## ✨ Tính năng chính

- **🎥 Phân tích Pose 3D** từ video đầu vào
- **🔄 So sánh Video** với video tham chiếu
- **📊 Tính toán độ chính xác** với điểm số chi tiết
- **🎨 Tạo video minh họa** với skeleton và màu sắc feedback
- **⚡ Xử lý Real-time** với tiến trình trực quan
- **🗂️ Quản lý Video tham chiếu** dễ dàng
- **📱 WebSocket Live Comparison** cho tương tác thời gian thực

## 🚀 Cài đặt và Khởi chạy

### Yêu cầu hệ thống
- Python 3.8+
- FFmpeg (để xử lý video)
- OpenCV và MediaPipe (để phân tích pose)

### Bước 1: Cài đặt dependencies

```bash
# Clone repository
git clone <repository-url>
cd PoseEstimation3D

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### Bước 2: Chuẩn bị dữ liệu

1. **Tạo thư mục cần thiết:**
```bash
mkdir -p uploads res/output reference_videos
```

2. **Thêm video tham chiếu** vào thư mục `reference_videos/`

### Bước 3: Khởi động server

```bash
# Khởi động server với reload tự động
uvicorn main_api:app --reload --host 0.0.0.0 --port 8000

# Hoặc chạy trực tiếp
python main_api.py
```

Server sẽ chạy tại: **http://127.0.0.1:8000**

## 📖 Tài liệu API

Truy cập giao diện Swagger UI để xem tài liệu chi tiết:
- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

---

## 🔧 Các API Endpoints

### 1. 📹 So sánh Video - `/api/compare_videos/`

**Endpoint chính để so sánh hai video và nhận kết quả chi tiết.**

#### Request
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Body:**
  - `user_video`: Video của người dùng (mp4, mov, avi)
  - `reference_video`: Video tham chiếu (mp4, mov, avi)

#### Response (Server-Sent Events)

##### Sự kiện `progress`
```json
{
  "step": "processing_frames",
  "message": "Processed frame 150/300 (50%)",
  "percentage": 50
}
```

##### Sự kiện `result`
```json
{
  "side_by_side_video_url": "/res/output/comparison_abc123.mp4",
  "annotated_user_video_url": "/res/output/annotated_user_video.mp4",
  "overall_accuracy": 85.5,
  "total_frames_processed": 300,
  "message": "Video comparison completed successfully"
}
```

#### Ví dụ JavaScript

```javascript
async function compareVideos(userFile, referenceFile) {
    const formData = new FormData();
    formData.append('user_video', userFile);
    formData.append('reference_video', referenceFile);

    const response = await fetch('http://127.0.0.1:8000/api/compare_videos/', {
        method: 'POST',
        body: formData
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
            if (line.startsWith('event: ')) {
                const event = line.replace('event: ', '');
                // Xử lý event tương ứng
                if (event === 'progress') {
                    // Cập nhật progress bar
                } else if (event === 'result') {
                    // Hiển thị kết quả
                }
            }
        }
    }
}
```

### 2. 🎯 Phân tích Performance - `/api/analyze_performance/`

**Phân tích video người dùng so với video tham chiếu có sẵn.**

#### Request
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Body:**
  - `user_video`: Video người dùng
  - `reference_video_id`: ID của video tham chiếu (tùy chọn)
  - `reference_video`: Upload video tham chiếu mới (tùy chọn)

### 3. 📚 Quản lý Reference Videos

#### Upload video tham chiếu
```http
POST /api/reference_videos/
Content-Type: multipart/form-data

file: [video file]
```

#### Lấy danh sách video tham chiếu
```http
GET /api/reference_videos/
```

### 4. ⚡ WebSocket Live Comparison

**So sánh thời gian thực qua WebSocket.**

```javascript
const ws = new WebSocket('ws://127.0.0.1:8000/ws/compare_live/{reference_video_id}');

// Gửi frames từ camera
const canvas = document.createElement('canvas');
// ... capture camera frame ...
ws.send(frameData);

// Nhận kết quả phân tích
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'comparison_result') {
        // Hiển thị score và keypoints
        console.log('Score:', data.score);
        console.log('Wrong keypoints:', data.wrong_keypoints);
    }
};
```

---

## 📁 Cấu trúc thư mục

```
PoseEstimation3D/
├── main_api.py           # API chính
├── pose_comparison.py    # Logic xử lý pose
├── api_models.py         # Models cho API
├── requirements.txt      # Dependencies
├── uploads/              # Video tạm thời (auto cleanup)
├── res/
│   └── output/          # Video kết quả (được giữ lại)
└── reference_videos/    # Video tham chiếu
```

## 🔒 Xử lý File

### File tạm thời
- **Lưu tại:** `uploads/`
- **Vòng đời:** Tự động xóa sau khi xử lý xong
- **Mục đích:** Chỉ để xử lý, không giữ lại

### File kết quả
- **Lưu tại:** `res/output/`
- **Vòng đời:** **Được giữ lại** để người dùng xem
- **Bao gồm:**
  - Video so sánh side-by-side
  - Video phân tích với skeleton màu
  - Các file metadata khác

### Video tham chiếu
- **Lưu tại:** `reference_videos/`
- **Vòng đời:** Giữ vĩnh viễn cho các lần sử dụng sau

## 🎨 Cách sử dụng Frontend

### 1. Upload và hiển thị progress

```html
<form id="upload-form">
    <input type="file" id="user_video" accept="video/*" required>
    <input type="file" id="reference_video" accept="video/*" required>
    <button type="submit">So sánh Videos</button>
</form>

<div id="progress-container" style="display: none;">
    <div id="progress-text">Đang xử lý...</div>
    <div id="progress-bar" style="width: 0%"></div>
</div>

<div id="result"></div>

<script>
// Xử lý form submit và SSE
</script>
```

### 2. Hiển thị kết quả

```javascript
// Khi nhận được result event
if (event === 'result') {
    const data = JSON.parse(eventData);

    result.innerHTML = `
        <h3>🎉 So sánh hoàn thành!</h3>
        <div style="display: flex; gap: 20px;">
            <div>
                <h4>📹 Video So sánh</h4>
                <video src="${data.side_by_side_video_url}" controls></video>
            </div>
            <div>
                <h4>🎨 Video Phân tích</h4>
                <video src="${data.annotated_user_video_url}" controls></video>
            </div>
        </div>
        <p><strong>Độ chính xác:</strong> ${data.overall_accuracy}%</p>
    `;
}
```

## 🛠️ Phát triển và Debug

### Logs hữu ích
- Các video output được log chi tiết đường dẫn
- Thông báo cleanup rõ ràng
- Progress updates chi tiết

### Debug mode
```bash
# Chạy với verbose logging
python -m uvicorn main_api:app --reload --log-level debug
```

## 📋 Ví dụ hoàn chỉnh

Xem file `test_comparison_api.html` để có ví dụ frontend hoàn chỉnh sử dụng API này.

## 🔄 Version History

- **v2.0:** Thêm video comparison, accuracy scoring, progress tracking
- **v1.0:** Basic 3D pose estimation

## 📞 Liên hệ

Nếu có vấn đề hoặc cần hỗ trợ, vui lòng tạo issue trong repository này.

---

**Chúc bạn sử dụng API vui vẻ! 🎯✨**