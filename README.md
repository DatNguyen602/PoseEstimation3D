# API Ước tính 3D Pose

Đây là backend API cho ứng dụng ước tính 3D pose từ video. API được xây dựng bằng FastAPI.

## 1. Cài đặt và Khởi chạy

### Yêu cầu
- Python 3.8+
- `pip`

### Bước 1: Cài đặt các thư viện cần thiết

Clone repository này và chạy lệnh sau trong thư mục gốc của dự án để cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### Bước 2: Khởi chạy Server

Sau khi cài đặt xong, sử dụng lệnh sau để khởi động API server:

```bash
uvicorn main_api:app --reload
```

Server sẽ chạy tại địa chỉ `http://127.0.0.1:8000`.

## 2. Cách Test và Sử dụng API

Cách dễ nhất để test API là sử dụng giao diện tài liệu tương tác (Swagger UI) do FastAPI tự động tạo ra.

#### 1. Video Comparison API (`/api/compare_videos/`) ⭐ **Khuyến nghị sử dụng**

Endpoint chính để so sánh 2 video và tạo video kết quả side-by-side với tiến độ real-time.

Đây là endpoint chính và được khuyến khích sử dụng. Nó cho phép upload 2 video (user video và reference video) và nhận về tiến trình xử lý real-time cũng như video kết quả cuối cùng thông qua một luồng Server-Sent Events.

- **Method**: `POST`
- **URL**: `http://127.0.0.1:8000/api/compare_videos/`
- **Body**: `multipart/form-data`
  - **Key**: `user_video` - File video của người dùng (mp4, mov, avi)
  - **Key**: `reference_video` - File video tham chiếu (mp4, mov, avi)

#### Phản hồi (Response)

API trả về một luồng **Server-Sent Events (SSE)**. Frontend cần lắng nghe các sự kiện sau:

1.  **Event: `progress`**
    - **Data**: JSON chứa thông tin tiến độ xử lý
    ```json
    {
      "type": "progress",
      "step": "processing_frames",
      "message": "Processed frame 150/300 (50%)",
      "percentage": 50
    }
    ```

2.  **Event: `result`**
    - **Data**: JSON chứa thông tin video kết quả
    ```json
    {
      "side_by_side_video_path": "/path/to/comparison_video.mp4",
      "side_by_side_video_url": "http://127.0.0.1:8000/res/output/comparison_abc123.mp4",
      "message": "Video comparison completed successfully"
    }
    ```

3.  **Event: `error`**
    - **Data**: Thông báo lỗi dạng string

4.  **Event: `done`**
    - **Data**: Thông báo hoàn thành xử lý

## 3. API Endpoints

### 3.1 Video Comparison API

**Endpoint:** `POST /api/compare_videos/`

#### Request
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Body:**
  - `user_video`: File video của người dùng (mp4, mov, avi)
  - `reference_video`: File video tham chiếu (mp4, mov, avi)

#### Response (Server-Sent Events)
API trả về luồng SSE với các sự kiện sau:

1. **Event: `progress`**
   ```json
   {
     "type": "progress",
     "step": "processing_frames",
     "message": "Processed frame 150/300 (50%)",
     "percentage": 50
   }
   ```

2. **Event: `result`** (khi hoàn thành)
   ```json
   {
     "side_by_side_video_path": "/path/to/comparison_video.mp4",
     "side_by_side_video_url": "http://127.0.0.1:8000/res/output/comparison_abc123.mp4",
     "message": "Video comparison completed successfully"
   }
   ```

3. **Event: `error`** (nếu có lỗi)
   ```json
   {
     "type": "error",
     "data": "Error message"
   }
   ```

#### Ví dụ JavaScript Frontend

```javascript
// Hàm upload và xử lý video comparison
async function compareVideos(userVideoFile, referenceVideoFile) {
    const formData = new FormData();
    formData.append('user_video', userVideoFile);
    formData.append('reference_video', referenceVideoFile);

    try {
        const response = await fetch('http://127.0.0.1:8000/api/compare_videos/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6); // Remove 'data: '

                    if (data.trim()) {
                        try {
                            const parsedData = JSON.parse(data);

                            if (parsedData.type === 'progress') {
                                // Cập nhật progress bar
                                updateProgressBar(parsedData.percentage, parsedData.message);
                            } else if (parsedData.type === 'result') {
                                // Hiển thị video kết quả
                                showResultVideo(parsedData.side_by_side_video_url);
                            } else if (parsedData.type === 'error') {
                                showError(parsedData.data);
                            }
                        } catch (e) {
                            // Xử lý log text thông thường
                            console.log('Log:', data);
                        }
                    }
                }
            }
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Không thể kết nối đến server');
    }
}

// Hàm cập nhật progress bar
function updateProgressBar(percentage, message) {
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');

    progressBar.style.width = percentage + '%';
    progressText.textContent = `${percentage}% - ${message}`;
}

// Hàm hiển thị video kết quả
function showResultVideo(videoUrl) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `
        <h3>🎉 So sánh hoàn thành!</h3>
        <video src="${videoUrl}" controls autoplay width="800"></video>
        <p><a href="${videoUrl}" target="_blank">Mở video trong tab mới</a></p>
    `;
}

// Hàm hiển thị lỗi
function showError(errorMessage) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `<div class="error">❌ Lỗi: ${errorMessage}</div>`;
}
```

#### Ví dụ HTML Form

```html
<form id="comparison-form">
    <div>
        <label>Video của bạn:</label>
        <input type="file" id="user-video" accept="video/*" required>
    </div>
    <div>
        <label>Video tham chiếu:</label>
        <input type="file" id="reference-video" accept="video/*" required>
    </div>
    <button type="submit">So sánh Videos</button>
</form>

<div id="progress-container" style="display: none;">
    <div id="progress-text">Đang xử lý...</div>
    <div style="background: #eee; height: 20px; border-radius: 10px;">
        <div id="progress-bar" style="background: #007bff; height: 100%; width: 0%; border-radius: 10px; transition: width 0.3s;"></div>
    </div>
</div>

<div id="result"></div>

<script>
// Thêm event listener cho form
document.getElementById('comparison-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const userVideo = document.getElementById('user-video').files[0];
    const referenceVideo = document.getElementById('reference-video').files[0];

    if (userVideo && referenceVideo) {
        // Hiển thị progress bar
        document.getElementById('progress-container').style.display = 'block';

        // Gọi API
        await compareVideos(userVideo, referenceVideo);
    }
});
</script>
```

#### Lưu ý
- **Progress tracking**: API cung cấp tiến độ real-time với percentage
- **Video URL**: Luôn sử dụng `side_by_side_video_url` thay vì tự tạo URL
- **Error handling**: Luôn xử lý lỗi và hiển thị cho người dùng
- **CORS**: API đã được cấu hình để cho phép truy cập từ frontend


source /home/minhdao/projects/Word/NCKH/PoseEstimation3D/.venv/bin/activate