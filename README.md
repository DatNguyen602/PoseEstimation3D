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

**Truy cập vào: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

### Endpoint chính: `/process-video-stream/`

Đây là endpoint chính và được khuyến khích sử dụng. Nó cho phép upload video và nhận về tiến trình xử lý cũng như kết quả cuối cùng thông qua một luồng (stream) sự kiện.

- **Method**: `POST`
- **URL**: `http://127.0.0.1:8000/process-video-stream/`
- **Body**: `multipart/form-data`
  - **Key**: `file`
  - **Value**: File video bạn muốn xử lý (định dạng .mp4, .mov, .avi).

#### Phản hồi (Response)

API sẽ trả về một luồng **Server-Sent Events (SSE)**. Frontend cần lắng nghe các sự kiện sau:

1.  **Event: `log`**
    - **Data**: Một chuỗi (string) chứa thông tin về tiến trình xử lý. Bạn có thể hiển thị trực tiếp các log này cho người dùng.
    ```
data: --- BƯỚC 1: PHÁT HIỆN 2D POSE ---
```

2.  **Event: `result`**
    - **Data**: Một chuỗi JSON chứa kết quả 3D pose cuối cùng. Đây là sự kiện báo hiệu xử lý thành công.
    ```json
data: {"poses_3d": [...], "metadata": {...}}
```

3.  **Event: `error`**
    - **Data**: Một chuỗi chứa thông báo lỗi nếu có sự cố xảy ra trong quá trình xử lý.

4.  **Event: `done`**
    - **Data**: Thông báo rằng quá trình xử lý đã kết thúc (dù thành công hay thất bại).

#### Ví dụ JavaScript (sử dụng `fetch`)

Đây là một đoạn mã ví dụ về cách frontend có thể gọi và xử lý luồng SSE từ API này.

```javascript
async function processVideo(videoFile) {
    const formData = new FormData();
    formData.append('file', videoFile);

    try {
        const response = await fetch('http://127.0.0.1:8000/process-video-stream/', {
            method: 'POST',
            body: formData
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let currentEvent = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                console.log("Stream finished.");
                break;
            }

            const chunk = decoder.decode(value);
            const lines = chunk.split("\n").filter(line => line.trim() !== '');

            for (const line of lines) {
                if (line.startsWith("event: ")) {
                    currentEvent = line.substring(7).trim();
                } else if (line.startsWith("data: ")) {
                    const data = line.substring(6);
                    handleEvent(currentEvent, data);
                }
            }
        }
    } catch (error) {
        console.error('Error processing video:', error);
    }
}

function handleEvent(eventName, data) {
    if (eventName === 'log') {
        console.log('LOG:', data);
        // Cập nhật UI với tiến trình
    } else if (eventName === 'result') {
        console.log('RESULT:', JSON.parse(data));
        // Hiển thị kết quả cuối cùng
    } else if (eventName === 'error') {
        console.error('ERROR:', data);
        // Hiển thị lỗi cho người dùng
    } else if (eventName === 'done') {
        console.log('DONE: Processing complete.');
    }
}
```