import os
import uuid
import shutil
import json
import uvicorn
import sys
import queue
import threading
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from sse_starlette.sse import EventSourceResponse

# Import the pipeline function
from run_pipeline import run_full_pipeline

# --- Cấu hình thư mục ---
UPLOADS_DIR = "uploads"
OUTPUTS_DIR = "res/output"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = FastAPI(
    title="3D Pose Estimation API",
    description="API để upload video và nhận lại dữ liệu 3D pose, hỗ trợ streaming tiến độ.",
    version="1.1.0"
)

# --- Lớp tiện ích để bắt output từ pipeline ---
class QueueIO:
    """Một lớp giả-file để ghi stdout vào một queue."""
    def __init__(self, q):
        self.q = q
    def write(self, s):
        # Chỉ đưa các chuỗi không rỗng vào queue
        if s.strip():
            self.q.put(s)
    def flush(self):
        pass

# --- Hàm chạy pipeline trong một thread riêng ---
def run_pipeline_in_thread(video_path, output_dir, output_basename, result_queue):
    """Chạy pipeline và chuyển hướng stdout vào queue."""
    try:
        # Chuyển hướng stdout
        original_stdout = sys.stdout
        sys.stdout = QueueIO(result_queue)
        
        # Chạy pipeline
        result = run_full_pipeline(video_path, output_dir, output_basename)
        
        # Đưa kết quả cuối cùng vào queue
        result_queue.put({"type": "result", "data": result})

    except Exception as e:
        import traceback
        error_str = traceback.format_exc()
        result_queue.put({"type": "error", "data": error_str})
    finally:
        # Khôi phục stdout và báo hiệu kết thúc
        sys.stdout = original_stdout
        result_queue.put({"type": "done"})

# --- Các hàm tiện ích cho API ---
def cleanup_files(files: list):
    """Xóa một danh sách các file."""
    print(f"🧹 Bắt đầu dọn dẹp {len(files)} file... ")
    for file_path in files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ Đã xóa: {file_path}")
        except Exception as e:
            print(f"⚠️ Lỗi khi dọn dẹp file {file_path}: {e}")

async def save_upload_file(upload_file: UploadFile) -> str:
    """Lưu file upload và trả về đường dẫn."""
    request_id = str(uuid.uuid4())
    _, extension = os.path.splitext(upload_file.filename)
    if extension.lower() not in ['.mp4', '.mov', '.avi']:
        raise HTTPException(status_code=400, detail="Định dạng file không hợp lệ.")
    
    video_filename = f"{request_id}{extension}"
    video_path = os.path.join(UPLOADS_DIR, video_filename)

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()
    
    print(f"📹 Đã lưu video vào: {video_path}")
    return video_path, request_id

# --- Endpoints API ---

@app.post("/process-video-stream/", 
          summary="Upload video và nhận tiến trình xử lý theo thời gian thực (SSE)")
async def process_video_stream(file: UploadFile = File(...) ):
    """    
    Upload video, chạy pipeline và stream tiến độ xử lý về cho client.
    - Client nhận các "log" event trong quá trình xử lý.
    - Event "result" cuối cùng sẽ chứa dữ liệu JSON 3D.
    - Event "error" sẽ được gửi nếu có lỗi xảy ra.
    """
    video_path, request_id = await save_upload_file(file)
    
    async def event_generator():
        result_queue = queue.Queue()
        files_to_cleanup = [video_path]

        # Chạy pipeline trong một thread khác để không block server
        pipeline_thread = threading.Thread(
            target=run_pipeline_in_thread,
            args=(video_path, OUTPUTS_DIR, request_id, result_queue)
        )
        pipeline_thread.start()

        try:
            while True:
                try:
                    # Lấy message từ queue (không blocking)
                    message = result_queue.get_nowait()
                    
                    if isinstance(message, dict):
                        # Xử lý các message đặc biệt (done, result, error)
                        if message["type"] == "done":
                            yield {"event": "done", "data": "Processing finished."}
                            break
                        elif message["type"] == "result":
                            final_json_path, generated_files = message["data"]
                            files_to_cleanup.extend(generated_files)
                            with open(final_json_path, 'r') as f:
                                json_data = json.load(f)
                            yield {"event": "result", "data": json.dumps(json_data)}
                        elif message["type"] == "error":
                            yield {"event": "error", "data": message["data"]}
                            break
                    else:
                        # Gửi log tiến trình
                        yield {"event": "log", "data": message}

                except queue.Empty:
                    # Nếu queue rỗng, đợi một chút và kiểm tra thread còn sống không
                    if not pipeline_thread.is_alive():
                        break
                    await asyncio.sleep(0.1)
        finally:
            # Đảm bảo thread kết thúc và dọn dẹp file
            pipeline_thread.join()
            cleanup_files(files_to_cleanup)

    return EventSourceResponse(event_generator())

@app.post("/process-video/", 
          summary="Upload và xử lý video (Request/Response đơn giản)",
          deprecated=True) # Đánh dấu là cũ, khuyến khích dùng stream
async def process_video_simple(background_tasks: BackgroundTasks, file: UploadFile = File(...) ):
    video_path, request_id = await save_upload_file(file)
    files_to_cleanup = [video_path]
    try:
        result = run_full_pipeline(video_path, OUTPUTS_DIR, request_id)
        if result is None:
            raise HTTPException(status_code=500, detail="Xử lý video thất bại.")
        final_json_path, generated_files = result
        files_to_cleanup.extend(generated_files)
        with open(final_json_path, 'r') as f:
            json_data = json.load(f)
        background_tasks.add_task(cleanup_files, files_to_cleanup)
        return json_data
    except Exception as e:
        background_tasks.add_task(cleanup_files, files_to_cleanup)
        raise HTTPException(status_code=500, detail=f"Lỗi nội bộ: {e}")

@app.get("/", summary="API Root", include_in_schema=False)
def read_root():
    return {"message": "Welcome to the 3D Pose Estimation API. Truy cập /docs để xem tài liệu."}

if __name__ == "__main__":
    print("🚀 Khởi động FastAPI server...")
    print("Truy cập http://127.0.0.1:8000/docs để xem giao diện API tương tác.")
    uvicorn.run(app, host="127.0.0.1", port=8000)