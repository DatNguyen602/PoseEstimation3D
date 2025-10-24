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


source /home/minhdao/projects/Word/NCKH/PoseEstimation3D/.venv/bin/activate

Server sẽ chạy tại địa chỉ `http://127.0.0.1:8000`.

## 2. Cách Test và Sử dụng API


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

## 4. Tích hợp Google Sheets (Tự động ghi kết quả)

API đã được tích hợp với Google Sheets để tự động ghi kết quả xử lý vào Google Spreadsheet sau mỗi lần chạy.

### 4.1 Thiết lập Google Sheets API

**Bước 1: Tạo Google Service Account**
1. Truy cập [Google Cloud Console](https://console.cloud.google.com/)
2. Tạo dự án mới hoặc chọn dự án hiện có
3. Bật Google Sheets API
4. Tạo Service Account và tải file `credentials.json`

**Bước 2: Cấu hình file credentials.json**
- Đặt file `credentials.json` vào thư mục gốc của dự án
- Đảm bảo file này được thêm vào `.gitignore`

**Bước 3: Tạo hoặc cấu hình Google Spreadsheet**
1. Tạo Google Spreadsheet mới hoặc sử dụng spreadsheet hiện có
2. Copy ID của spreadsheet từ URL (phần sau `/d/` và trước `/edit`)
   - Ví dụ: `https://docs.google.com/spreadsheets/d/1tmJEirxVYFdvdJxA8tfMqX5fcWzRxtGv0WHV1NRNcxY/edit`
   - Spreadsheet ID: `1tmJEirxVYFdvdJxA8tfMqX5fcWzRxtGv0WHV1NRNcxY`

**Bước 4: Cập nhật file .env**
```bash
# Google Sheets Integration Configuration
GOOGLE_APPLICATION_CREDENTIALS=credentials.json
GOOGLE_SHEET_ID=your_spreadsheet_id_here

# Ví dụ: GOOGLE_SHEET_ID=1tmJEirxVYFdvdJxA8tfMqX5fcWzRxtGv0WHV1NRNcxY
```

### 4.2 Cấu trúc dữ liệu trong Google Sheets

Sau khi tích hợp, Google Sheets sẽ có 4 sheets:

#### Sheet "Tổng quan"
- **Thời gian**: Thời gian xử lý
- **User ID**: ID người dùng (nếu có)
- **Video người dùng**: Tên file video người dùng
- **Điểm tương đồng**: Điểm tương đồng trung bình (%)
- **Điểm nhịp**: Điểm nhịp điệu (%)
- **Điểm tư thế**: Điểm tư thế (%)
- **Điểm động tác**: Điểm động tác (%)
- **Điểm biểu cảm**: Điểm biểu cảm (%)
- **Tổng điểm**: Tổng điểm khiêu vũ (%)
- **Baseline Score**: Điểm baseline (%)
- **Sensitivity Score**: Điểm sensitivity (%)
- **Frames processed**: Số frames đã xử lý
- **Cloudinary URL**: URL video trên Cloudinary
- **Excel URL**: URL file Excel kết quả
- **Database Record ID**: ID bản ghi trong database

#### Sheet "Chi tiết frame"
- **Frame**: Số thứ tự frame
- **Timestamp**: Thời gian của frame
- **Similarity Score**: Điểm tương đồng của frame
- **Rhythm Score**: Điểm nhịp điệu của frame
- **Posture Score**: Điểm tư thế của frame
- **Movement Score**: Điểm động tác của frame
- **Expression Score**: Điểm biểu cảm của frame

#### Sheet "Mophong"
- **Chuannhip**: Điểm chuẩn nhịp được tính từ điểm tổng với biến thiên ±10%
- **Tuthe**: Điểm tư thế được tính từ điểm tổng với biến thiên ±8%
- **Dongtactay**: Điểm động tác tay được tính từ điểm tổng với biến thiên ±12%
- **Bieucam**: Điểm biểu cảm được tính từ điểm tổng với biến thiên ±6%
- **Diemtong**: Điểm tổng được random trong khoảng 70-95 điểm

### 4.3 Cách hoạt động

1. **Tự động ghi**: Sau mỗi lần xử lý video thành công, API sẽ tự động ghi kết quả vào Google Sheets
2. **Lỗi cũng được ghi**: Nếu có lỗi xảy ra, thông tin lỗi cũng được ghi vào sheet để theo dõi
3. **Không ảnh hưởng xử lý**: Việc ghi vào Google Sheets không ảnh hưởng đến quá trình xử lý chính
4. **Real-time**: Dữ liệu được ghi ngay sau khi xử lý xong

### 4.4 Ví dụ response với Google Sheets status

Khi xử lý hoàn thành, response sẽ bao gồm trạng thái Google Sheets:

```json
{
  "type": "completed",
  "message": "Video comparison completed successfully",
  "result_url": "https://res.cloudinary.com/...",
  "excel_url": "/res/output/dance_scoring_abc123.xlsx",
  "average_score": 85.5,
  "dance_metrics": {
    "rhythm_score": 82.0,
    "posture_score": 88.0,
    "movement_score": 84.0,
    "expression_score": 87.0,
    "total_score": 85.25
  },
  "google_sheets_status": "success"
}
```

### 4.5 Troubleshooting

**Lỗi thường gặp:**
- `"google_sheets_status": "disabled"` - Chưa cấu hình Google Sheets API
- Lỗi authentication - Kiểm tra file `credentials.json`
- Lỗi permission - Đảm bảo Service Account có quyền ghi vào spreadsheet

**Các lỗi phổ biến và cách khắc phục:**

#### 1. **403 Forbidden - Không có quyền truy cập**
**Nguyên nhân:** Service Account không được chia sẻ quyền truy cập spreadsheet

**Cách khắc phục:**
1. Mở Google Spreadsheet mà bạn muốn ghi dữ liệu vào
2. Click vào nút **"Share"** (Chia sẻ) ở góc trên bên phải
3. Nhập email Service Account: **`sheet-video-3d@main-nucleus-475706-a2.iam.gserviceaccount.com`**
4. Chọn quyền **`Editor`** (Người chỉnh sửa)
5. Click **"Share"**

**Ví dụ hình ảnh:**
```
Email: sheet-video-3d@main-nucleus-475706-a2.iam.gserviceaccount.com
Quyền: Editor
```

#### 2. **404 Not Found - Không tìm thấy spreadsheet**
**Nguyên nhân:** Spreadsheet ID không đúng hoặc spreadsheet đã bị xóa

**Cách khắc phục:**
1. Kiểm tra lại URL của Google Spreadsheet
2. Copy đúng phần ID từ URL: `https://docs.google.com/spreadsheets/d/[SPREADSHEET_ID]/edit`
3. Cập nhật lại `GOOGLE_SHEET_ID` trong file `.env`

#### 3. **Authentication Failed**
**Nguyên nhân:** File `credentials.json` không đúng hoặc bị thiếu

**Cách khắc phục:**
1. Kiểm tra file `credentials.json` có tồn tại không
2. Đảm bảo nội dung file đúng với thông tin Service Account đã cung cấp
3. Nếu bị thiếu, tạo lại Service Account mới trong Google Cloud Console

#### 5. **400 Bad Request - Không thể parse range**
**Nguyên nhân:** Tên sheet có dấu tiếng Việt hoặc ký tự đặc biệt gây lỗi parse range

**Cách khắc phục:**
1. Đổi tên các sheet hiện tại từ có dấu sang không dấu:
   - `'Tổng quan'` → `'Tongquan'`
   - `'Chi tiết frame'` → `'Chitietframe'`
   - `'Thông số kỹ thuật'` → `'Thongsokythuat'`
2. Hoặc tạo spreadsheet mới với tên sheet không dấu ngay từ đầu

**Ví dụ đổi tên sheet:**
```
1. Click chuột phải vào tab sheet
2. Chọn "Rename" 
3. Đổi từ "Tổng quan" thành "Tongquan"
4. Lặp lại với các sheet khác
```

### 5.2 List Reference Videos API

**Endpoint:** `GET /api/reference_videos/`

Lấy danh sách tất cả reference videos đã upload.

#### Response
```json
[
  {
    "video_id": "abc123def456"
  },
  {
    "video_id": "xyz789ghi012"
  }
]
```

### 5.3 Process Video Stream API

**Endpoint:** `POST /process-video-stream/`

Xử lý video đơn lẻ để ước tính 3D pose (không so sánh).

#### Request
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Body:**
  - `file`: File video cần xử lý
  - `user_id`: (Optional) ID người dùng
  - `title`: (Optional) Tiêu đề

### 5.4 Database Integration APIs

#### Get Compare Videos Results
**Endpoint:** `GET /api/get_compare_videos_results/`

Lấy kết quả so sánh video từ database.

**Query Parameters:**
- `user_id`: (Optional) Lọc theo user ID
- `limit`: Số bản ghi tối đa (default: 50)
- `offset`: Bỏ qua số bản ghi (default: 0)

#### Get Video Stream Results
**Endpoint:** `GET /api/get_video_stream_results/`

Lấy kết quả xử lý video stream từ database.

**Query Parameters:**
- `user_id`: (Optional) Lọc theo user ID
- `limit`: Số bản ghi tối đa (default: 50)
- `offset`: Bỏ qua số bản ghi (default: 0)

## 6. Cấu trúc thư mục

```
PoseEstimation3D/
├── main_api.py              # File chính của API
├── google_sheets_helper.py  # Helper để tích hợp Google Sheets
├── credentials.json         # File credentials Google Service Account
├── requirements.txt         # Các thư viện cần thiết
├── .env                     # Cấu hình môi trường
├── database_manager.py      # Quản lý database
├── pose_comparison.py       # Logic xử lý pose comparison
├── dance_scoring.py         # Logic chấm điểm khiêu vũ
├── cloudinary_helper.py     # Helper upload lên Cloudinary
└── res/
    └── output/              # Thư mục chứa kết quả đầu ra
```

### Google Sheets Structure:
- **Tongquan**: Tổng quan kết quả với đầy đủ thông tin chi tiết
- **Chitietframe**: Chi tiết từng frame với điểm số cụ thể
- **Thongsokythuat**: Thông số kỹ thuật và metric đánh giá
- **Mophong**: Sheet đặc biệt với 5 cột đánh giá khiêu vũ (Chuannhip, Tuthe, Dongtactay, Bieucam, Diemtong) với logic tính điểm từ điểm tổng được random

## 8. Troubleshooting

### Lỗi thường gặp

{{ ... }}
```bash
pip install -r requirements.txt
```

**2. Lỗi Google Sheets API**
- Kiểm tra file `credentials.json`
- Đảm bảo đã bật Google Sheets API trong Google Cloud Console
- Kiểm tra quyền của Service Account

**3. Lỗi database**
- Kiểm tra kết nối database trong `database_manager.py`
- Đảm bảo bảng đã được tạo: `python -c "from database_manager import db_manager; db_manager.create_table_if_not_exists()"`

**4. Lỗi video processing**
- Kiểm tra đường dẫn đến các thư viện pose estimation
- Đảm bảo video có định dạng hỗ trợ (mp4, mov, avi)

### Logs và Debug

- File log chính: `api.log`
- Để xem logs real-time: `tail -f api.log`
- Để chạy với debug mode: thêm `--log-level debug` khi chạy uvicorn

## 9. Lệnh chạy nhanh

### Khởi chạy server
```bash
# Khởi chạy với reload (development)
uvicorn main_api:app --reload --host 0.0.0.0 --port 8000

# Khởi chạy production mode
uvicorn main_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Cài đặt thư viện
```bash
# Cài đặt tất cả dependencies
pip install -r requirements.txt

# Cài đặt riêng Google Sheets dependencies
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

### Test API
```bash
# Test endpoint cơ bản
curl -X GET "http://localhost:8000/api/reference_videos/"

# Test upload reference video
curl -X POST "http://localhost:8000/api/reference_videos/" \
     -F "file=@sample_video.mp4"
```

### Debug và Logs
```bash
# Xem logs real-time
tail -f api.log

# Lọc logs liên quan đến Google Sheets
tail -f api.log | grep -i "google sheets"

# Lọc logs lỗi
tail -f api.log | grep -i "error\|exception\|failed"
```

### Kiểm tra database
```bash
# Test kết nối database
python -c "from database_manager import db_manager; print(db_manager.test_connection())"

# Tạo bảng database nếu chưa có
python -c "from database_manager import db_manager; db_manager.create_table_if_not_exists()"
## 10. Phiên bản và Changelog

- **Version 2.1.0**: 
  - Thêm sheet "Mophong" với 5 cột đặc biệt và logic tính điểm từ điểm tổng được random
  - Điểm tổng được random trong khoảng 70-95 điểm
  - Các điểm thành phần được tính từ điểm tổng với biến thiên hợp lý (±6% đến ±12%)
  - Tối ưu hiệu suất xử lý video và Google Sheets integration

---

🎉 **Chúc bạn sử dụng API thành công!** Nếu có vấn đề gì, hãy kiểm tra logs và làm theo hướng dẫn troubleshooting ở trên.