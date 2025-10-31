import os
import uuid
import shutil
import json
import uvicorn
import sys
import queue
import threading
import asyncio
import traceback
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

print("Importing pose_comparison...")
from pose_comparison import PoseComparison, LiveComparisonSession, LiveCameraSession
print("Imported pose_comparison.")

# Import database manager
print("Importing database_manager...")
from database_manager import db_manager
print("Imported database_manager.")

# Import pipeline runner
print("Importing run_pipeline...")
from run_pipeline import run_full_pipeline
print("Imported run_pipeline.")

# Import Cloudinary helper
print("Importing cloudinary_helper...")
from cloudinary_helper import upload_comparison_video
print("Imported cloudinary_helper.")

# Import Google Sheets helper
print("Importing google_sheets_helper...")
from google_sheets_helper import GoogleSheetsHelper
print("Imported google_sheets_helper.")

# Import derived score calculator
from dance_scoring_calculator import DanceScoringCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

# --- Directory Configuration ---
UPLOADS_DIR = "uploads"
OUTPUTS_DIR = "res/output"
REFERENCE_VIDEOS_DIR = "reference_videos"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(REFERENCE_VIDEOS_DIR, exist_ok=True)

app = FastAPI(
    title="Pose Estimation & Comparison API",
    description="An advanced API for 3D pose estimation and performance comparison, featuring real-time feedback and video management.",
    version="2.0.0"
)

sheets_helper = None

@app.on_event("startup")
def startup_event():
    """
    Initializes helpers on application startup.
    """
    global sheets_helper
    logger.info("--- Running startup_event for Google Sheets initialization ---")
    try:
        sheets_helper = GoogleSheetsHelper()
        logger.info("✅ Google Sheets helper initialized successfully!")

        # Kiểm tra quyền truy cập spreadsheet nếu đã cấu hình
        if sheets_helper.spreadsheet_id and sheets_helper.spreadsheet_id != 'your_spreadsheet_id_here':
            access_ok = sheets_helper.check_spreadsheet_access()
            if not access_ok:
                logger.warning("⚠️ Không thể truy cập Google Sheets được cấu hình")
                logger.info("ℹ️ Hãy kiểm tra quyền chia sẻ spreadsheet với Service Account")
            else:
                logger.info("✅ Đã xác nhận quyền truy cập Google Sheets")
        else:
            logger.info("ℹ️ Google Sheets ID chưa được cấu hình - sẽ tạo spreadsheet mới khi cần")

    except Exception as e:
        logger.warning(f"⚠️ Google Sheets helper initialization failed: {e}")
        logger.info("ℹ️ Google Sheets integration will be disabled. No data will be written to sheets.")

# --- CORS Configuration ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False, # Set to False when using wildcard origin
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount static files directory for output videos
app.mount("/res/output", StaticFiles(directory=OUTPUTS_DIR), name="output")

# --- Utility Classes & Functions ---
class QueueIO:
    """A file-like class to write stdout to a queue."""
    def __init__(self, q):
        self.q = q
    def write(self, s):
        if s.strip():
            self.q.put(s)
    def flush(self):
        pass

def cleanup_files(files: list):
    """Deletes a list of files."""
    logger.info(f"🧹 Starting cleanup for {len(files)} files...")
    for file_path in files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"🗑️ Deleted: {file_path}")
        except Exception as e:
            logger.error(f"⚠️ Error cleaning up file {file_path}: {e}")

async def save_upload_file(upload_file: UploadFile) -> str:
    """Saves a temporary uploaded file and returns its path and request ID."""
    request_id = str(uuid.uuid4())
    _, extension = os.path.splitext(upload_file.filename)
    if extension.lower() not in ['.mp4', '.mov', '.avi', '.webm', '.mkv']:
        raise HTTPException(status_code=400, detail="Invalid file format.")
    
    video_filename = f"{request_id}{extension}"
    video_path = os.path.join(UPLOADS_DIR, video_filename)

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()
    
    logger.info(f"📹 Saved temporary video to: {video_path}")
    return video_path, request_id

# --- Reference Video Management ---

async def save_reference_video(upload_file: UploadFile) -> str:
    """Saves an uploaded reference video with a unique ID and returns the ID."""
    _, extension = os.path.splitext(upload_file.filename)
    if extension.lower() not in ['.mp4', '.mov', '.avi']:
        raise HTTPException(status_code=400, detail="Invalid video file format. Only .mp4, .mov, .avi are allowed.")
    
    video_id = f"{uuid.uuid4()}{extension}"
    video_path = os.path.join(REFERENCE_VIDEOS_DIR, video_id)

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()
    
    logger.info(f"📹 Saved new reference video to: {video_path}")
    return video_id

@app.post("/api/reference_videos/", 
          summary="Upload a new reference video",
          status_code=201)
async def upload_reference_video(file: UploadFile = File(...)):
    """
    Upload a video to be used as a reference for comparisons.
    The video is saved with a unique ID, which is returned in the response.
    This ID can then be used to start a live comparison session.
    """
    video_id = await save_reference_video(file)
    return {"video_id": video_id, "filename": file.filename}

@app.get("/api/reference_videos/", summary="List all available reference videos")
async def list_reference_videos():
    """
    Returns a list of all available reference videos that can be used for comparison.
    Each video is identified by a unique `video_id` (the filename).
    """
    try:
        files = os.listdir(REFERENCE_VIDEOS_DIR)
        # In the future, we could add more metadata here (duration, thumbnail, etc.)
        videos = [{"video_id": f} for f in files if f.lower().endswith(('.mp4', '.mov', '.avi'))]
        return videos
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read reference videos directory: {e}")

# --- Pose Estimation & Comparison Endpoints ---

def run_pipeline_in_thread(video_path, output_dir, output_basename, result_queue):
    """Runs the 3D pose estimation pipeline in a separate thread."""
    logger.info(f"🚀 Pipeline thread started for video: {video_path} -> {output_basename}")
    try:
        original_stdout = sys.stdout
        sys.stdout = QueueIO(result_queue)
        logger.info(f"▶️ Running full pipeline for video: {video_path}")
        result = run_full_pipeline(video_path, output_dir, output_basename)
        logger.info(f"✅ Pipeline completed successfully for video: {video_path}")
        result_queue.put({"type": "result", "data": result})
    except Exception as e:
        error_str = traceback.format_exc()
        logger.error(f"❌ Pipeline error for video {video_path}: {e}")
        result_queue.put({"type": "error", "data": error_str})
    finally:
        sys.stdout = original_stdout
        result_queue.put({"type": "done"})
        logger.info(f"🏁 Pipeline thread finished for video: {video_path}")

@app.post("/process-video-stream/", 
          summary="Upload video for 3D pose estimation (SSE)",
          tags=["Video Processing"])
async def process_video_stream(
    file: UploadFile = File(...),
    user_id: str = Form(None),
    title: str = Form(None)
):
    video_path, request_id = await save_upload_file(file)
    logger.info(
        "📥 Received process_video_stream request",
        extra={
            "request_id": request_id,
            "user_id": user_id,
            "title": title,
            "video_path": video_path
        }
    )
    
    async def event_generator():
        result_queue = queue.Queue()
        files_to_cleanup = [video_path]
        pipeline_thread = threading.Thread(
            target=run_pipeline_in_thread,
            args=(video_path, OUTPUTS_DIR, request_id, result_queue)
        )
        pipeline_thread.start()
        logger.info(f"🧵 Pipeline thread started (request_id={request_id})")
        try:
            logger.info(f"📡 SSE event generator running (request_id={request_id})")
            while True:
                try:
                    message = result_queue.get_nowait()
                    if isinstance(message, dict):
                        logger.info(f"📨 Received pipeline message: {message.get('type')} (request_id={request_id})")
                        if message["type"] == "done":
                            yield {"event": "done", "data": "Processing finished."}
                            break
                        elif message["type"] == "result":
                            # Process and save result
                            final_json_path, generated_files = message["data"]
                            files_to_cleanup.extend(generated_files)
                            
                            with open(final_json_path, 'r') as f:
                                result_data = json.load(f)
                            
                            result_data['input_video_url'] = file.filename
                            result_data['result_url'] = f"/res/output/{os.path.basename(final_json_path)}"
                            if user_id:
                                result_data['user_id'] = user_id
                            if title:
                                result_data['title'] = title
                            
                            try:
                                logger.info(f"💾 Saving result to database for user: {user_id}")
                                record_id = db_manager.save_video_result(
                                    result_data,
                                    'process_video_stream',
                                    user_id
                                )
                                result_data['database_record_id'] = record_id
                                logger.info(f"✅ Successfully saved to database with ID: {record_id}")
                            except Exception as db_error:
                                logger.error(f"❌ Failed to save result to database: {str(db_error)}")
                                logger.error(f"🔍 DB Error Details: {traceback.format_exc()}")
                                result_data['database_error'] = str(db_error)

                            yield {"event": "result", "data": json.dumps(result_data)}
                            
                        elif message["type"] == "error":
                            try:
                                logger.info(f"💾 Saving error to database for user: {user_id}")
                                error_data = {
                                    'error': message['data'],
                                    'input_video_url': file.filename,
                                    'title': title or 'Video Processing Failed'
                                }
                                if user_id:
                                    error_data['user_id'] = user_id
                                
                                db_manager.save_video_result(
                                    error_data,
                                    'process_video_stream',
                                    user_id
                                )
                                logger.info(f"✅ Successfully saved error to database")
                            except Exception as db_error:
                                logger.error(f"❌ Failed to save error to database: {str(db_error)}")
                                logger.error(f"🔍 DB Error Details: {traceback.format_exc()}")
                            
                            yield {"event": "error", "data": message["data"]}
                            break
                    else:
                        logger.debug(f"📝 Log stream message: {message.strip()} (request_id={request_id})")
                        yield {"event": "log", "data": message}
                except queue.Empty:
                    if not pipeline_thread.is_alive():
                        logger.info(f"📭 Queue empty and pipeline thread finished (request_id={request_id})")
                        break
                    await asyncio.sleep(0.1)
        finally:
            pipeline_thread.join()
            logger.info(f"🧼 Cleaning up files for request_id={request_id}: {files_to_cleanup}")
            cleanup_files(files_to_cleanup)
    return EventSourceResponse(event_generator())

def run_video_comparison_in_thread(user_video_path, reference_video_path, output_path, result_queue):
    """
    Runs the video comparison process in a thread.
    Creates side-by-side comparison video and gets synchronized scores.
    """
    try:
        logger.info(f"🔄 Starting video comparison process...")
        comparison = PoseComparison(reference_video_path)
        
        logger.info(f"⚙️ Processing video files and creating side-by-side comparison...")
        result_dict, frame_scores = comparison.process_video_files(user_video_path, output_path, result_queue)

        logger.info(f"✅ Video comparison completed successfully!")
        
        # Generate proper video URL for frontend
        video_filename = os.path.basename(output_path)
        video_url = f"/res/output/{video_filename}"

        # Construct the final result data object - scores are now already synchronized
        result_data = {
            "cloudinary_video_url": video_url,
            "result_url": video_url,
            "side_by_side_video_url": video_url,
            "average_similarity_score": result_dict.get('average_similarity_score', 0),
            "message": "Video comparison completed successfully",
            "total_frames_processed": result_dict.get('total_frames_processed', 0),
            "score_range": {
                "min": min(frame_scores) if frame_scores else 0,
                "max": max(frame_scores) if frame_scores else 0,
                "average": result_dict.get('average_similarity_score', 0)
            },
            "dance_scoring_metrics": result_dict.get('dance_scoring_metrics', {}),
            "frame_details": result_dict.get('frame_details', [])
        }
        
        result_queue.put({"type": "progress", "step": "completed", "message": "Comparison completed!", "percentage": 100})
        result_queue.put({"type": "result", "data": result_data})

    except Exception as e:
        error_str = traceback.format_exc()
        error_msg = f"❌ Error in video comparison: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"🔍 Debug info - User: {user_video_path}, Ref: {reference_video_path}")
        logger.debug(f"📋 Full traceback: {error_str}")
        result_queue.put({"type": "error", "data": error_msg})
    finally:
        logger.info(f"🏁 Video comparison thread finished")
        result_queue.put({"type": "done"})


def export_dance_scoring_to_excel(result_data: Dict[str, Any], request_id: str) -> str:
    """
    Xuất kết quả đánh giá khiêu vũ ra file Excel với 4 trường mới

    Args:
        result_data: Dữ liệu kết quả từ API
        request_id: ID của request để tạo tên file

    Returns:
        Đường dẫn file Excel
    """
    try:
        # Import pandas và các thư viện cần thiết
        import pandas as pd
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter

        excel_filename = f"dance_scoring_{request_id}.xlsx"
        excel_path = os.path.join(OUTPUTS_DIR, excel_filename)

        # Chuẩn bị dữ liệu cho các sheet
        overview_data = {
            'Tiêu chí': ['Điểm tương đồng', 'Chuẩn nhịp', 'Tư thế', 'Động tác', 'Biểu cảm', 'Tổng điểm'],
            'Điểm số': [
                result_data.get('average_similarity_score', 0),
                result_data.get('dance_scoring_metrics', {}).get('rhythm_score', 0),
                result_data.get('dance_scoring_metrics', {}).get('posture_score', 0),
                result_data.get('dance_scoring_metrics', {}).get('movement_score', 0),
                result_data.get('dance_scoring_metrics', {}).get('expression_score', 0),
                result_data.get('dance_scoring_metrics', {}).get('total_score', 0)
            ]
        }

        # Tạo DataFrame cho sheet tổng quan
        overview_df = pd.DataFrame(overview_data)

        # Chuẩn bị dữ liệu chi tiết từng frame
        frame_details = result_data.get('frame_details', [])
        if frame_details:
            details_df = pd.DataFrame(frame_details)
            # Làm tròn các điểm số để dễ đọc
            score_columns = ['similarity_score', 'rhythm_score', 'posture_score', 'movement_score', 'expression_score']
            for col in score_columns:
                if col in details_df.columns:
                    details_df[col] = details_df[col].round(2)
        else:
            details_df = pd.DataFrame()

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet Tổng quan với định dạng đẹp
            overview_df.to_excel(writer, sheet_name='Tổng quan', index=False, startrow=2)

            workbook = writer.book
            worksheet = writer.sheets['Tổng quan']

            # Thêm tiêu đề chính
            title_cell = worksheet.cell(row=1, column=1)
            title_cell.value = "BÁO CÁO ĐÁNH GIÁ BIỂU DIỄN KHIÊU VŨ"
            title_cell.font = Font(size=16, bold=True, color="000000")
            worksheet.merge_cells('A1:F1')

            # Căn giữa tiêu đề
            title_cell.alignment = Alignment(horizontal='center')

            # Định dạng header
            header_font = Font(bold=True, color="FFFFFF")
            header_fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")

            for col_num in range(1, len(overview_df.columns) + 1):
                cell = worksheet.cell(row=3, column=col_num)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal='center')

            # Định dạng các ô điểm số
            score_fill = PatternFill(start_color="E6F3FF", end_color="E6F3FF", fill_type="solid")
            for row_num in range(4, len(overview_df) + 4):
                for col_num in range(2, len(overview_df.columns) + 1):
                    cell = worksheet.cell(row=row_num, column=col_num)
                    cell.fill = score_fill
                    cell.alignment = Alignment(horizontal='center')

            # Điều chỉnh độ rộng cột
            for col_num in range(1, len(overview_df.columns) + 1):
                col_letter = get_column_letter(col_num)
                worksheet.column_dimensions[col_letter].width = 15

            # Sheet Chi tiết từng frame
            if not details_df.empty:
                details_df.to_excel(writer, sheet_name='Chi tiết frame', index=False)

                # Định dạng sheet chi tiết
                detail_sheet = writer.sheets['Chi tiết frame']
                detail_header_font = Font(bold=True, color="FFFFFF")
                detail_header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")

                # Định dạng header cho sheet chi tiết
                for col_num in range(1, len(details_df.columns) + 1):
                    cell = detail_sheet.cell(row=1, column=col_num)
                    cell.font = detail_header_font
                    cell.fill = detail_header_fill
                    cell.alignment = Alignment(horizontal='center')

                # Điều chỉnh độ rộng cột cho sheet chi tiết
                for col_num in range(1, len(details_df.columns) + 1):
                    col_letter = get_column_letter(col_num)
                    if col_num <= 3:  # Các cột đầu (Frame, Timestamp, Similarity)
                        detail_sheet.column_dimensions[col_letter].width = 12
                    else:  # Các cột điểm số
                        detail_sheet.column_dimensions[col_letter].width = 15

            # Sheet Thông số kỹ thuật
            technical_data = result_data.get('dance_scoring_metrics', {})
            if technical_data:
                tech_df = pd.DataFrame([{
                    'Baseline Score (%)': technical_data.get('baseline_score_percent', 0),
                    'Sensitivity Score (%)': technical_data.get('sensitivity_score_percent', 0),
                    'Baseline Processing Time (s)': technical_data.get('baseline_processing_time', 0),
                    'Sensitivity Processing Time (s)': technical_data.get('sensitivity_processing_time', 0),
                    'Total Frames Processed': result_data.get('total_frames_processed', 0),
                    'Processing Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }])
                tech_df.to_excel(writer, sheet_name='Thông số kỹ thuật', index=False)

        logger.info(f"✅ Dance scoring Excel exported successfully: {excel_path}")
        return excel_path

    except ImportError as e:
        logger.error(f"❌ Missing required libraries for Excel export: {e}")
        raise HTTPException(status_code=500, detail="Excel export not available - missing pandas/openpyxl")
    except Exception as e:
        logger.error(f"❌ Error exporting dance scoring to Excel: {e}")
        raise HTTPException(status_code=500, detail=f"Excel export failed: {str(e)}")

    except Exception as e:
        error_str = traceback.format_exc()
        error_msg = f"❌ Error in video comparison: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"🔍 Debug info - User: {user_video_path}, Ref: {reference_video_path}")
        logger.debug(f"📋 Full traceback: {error_str}")
        result_queue.put({"type": "error", "data": error_msg})
    finally:
        logger.info(f"🏁 Video comparison thread finished")
        result_queue.put({"type": "done"})

@app.post("/api/compare_videos/", 
          summary="Compare two uploaded videos (batch processing)",
          tags=["Video Comparison"])
async def compare_videos(user_video: UploadFile = File(...), reference_video: UploadFile = File(...), save_to_db: bool = True, user_id: str = Form(None), title: str = Form(None)):
    """
    Compare two uploaded videos (batch processing).
    
    Args:
        user_video: The user's video file
        reference_video: The reference video file
        save_to_db: Whether to save results to database (default: True)
        user_id: User ID for database record (optional)
        title: Title for database record (optional)
    """
    user_video_path, user_request_id = await save_upload_file(user_video)
    ref_video_path, _ = await save_upload_file(reference_video)
    
    output_filename = f"comparison_{user_request_id}.mp4"
    output_path = os.path.join(OUTPUTS_DIR, output_filename) 

    user_video_basename = os.path.basename(user_video_path)
    annotated_filename = f"annotated_{user_video_basename}"
    annotated_output_path = os.path.join(OUTPUTS_DIR, annotated_filename)

    async def event_generator():
        result_queue = queue.Queue()
        # Only clean up the temporary uploaded files, not the results
        files_to_cleanup = [user_video_path, ref_video_path]
        comparison_thread = threading.Thread(
            target=run_video_comparison_in_thread,
            args=(user_video_path, ref_video_path, output_path, result_queue)
        )
        comparison_thread.start()
        try:
            while True:
                try:
                    message = result_queue.get_nowait()
                    if isinstance(message, dict):
                        if message["type"] == "done":
                            yield {"event": "done", "data": "Processing finished."}
                            break
                        elif message["type"] == "result":
                            # Return the final result with video URL
                            result_data = message["data"]
                            
                            # Upload video to Cloudinary for better streaming performance
                            cloudinary_upload_result = None
                            try:
                                logger.info(f"☁️ Đang upload video lên Cloudinary: {output_path}")
                                cloudinary_upload_result = upload_comparison_video(output_path, user_request_id)
                                
                                if cloudinary_upload_result.get('success'):
                                    logger.info(f"✅ Video uploaded to Cloudinary successfully: {cloudinary_upload_result['public_id']}")
                                    # Add Cloudinary URLs to result data
                                    result_data['cloudinary_video_url'] = cloudinary_upload_result['secure_url']
                                    result_data['cloudinary_public_id'] = cloudinary_upload_result['public_id']
                                    # Add adaptive streaming URL if available
                                    from cloudinary_helper import CloudinaryHelper
                                    adaptive_url = CloudinaryHelper.get_adaptive_streaming_url(cloudinary_upload_result['public_id'])
                                    if adaptive_url:
                                        result_data['cloudinary_streaming_url'] = adaptive_url
                                        logger.info(f"📺 Adaptive streaming URL: {adaptive_url}")
                                else:
                                    logger.error(f"❌ Failed to upload to Cloudinary: {cloudinary_upload_result.get('error')}")
                                    # Continue with local URL even if Cloudinary upload fails
                            except Exception as cloudinary_error:
                                logger.error(f"❌ Cloudinary upload error: {str(cloudinary_error)}")
                                # Continue with local URL even if Cloudinary upload fails
                            
                            # Save to database if requested
                            if save_to_db:
                                try:
                                    logger.info(f"💾 Saving result to database for user: {user_id}")
                                    result_data['input_video_url'] = user_video.filename
                                    if user_id:
                                        result_data['user_id'] = user_id
                                    if title:
                                        result_data['title'] = title
                                    
                                    if 'cloudinary_video_url' in result_data:
                                        result_data['result_url'] = result_data['cloudinary_video_url']
                                    
                                    record_id = db_manager.save_video_result(
                                        result_data, 
                                        'compare_videos',
                                        user_id
                                    )
                                    
                                    result_data['database_record_id'] = record_id
                                    logger.info(f"✅ Successfully saved to database with ID: {record_id}")
                                    
                                except Exception as db_error:
                                    logger.error(f"❌ Failed to save result to database: {str(db_error)}")
                                    logger.error(f"🔍 DB Error Details: {traceback.format_exc()}")
                                    result_data['database_error'] = str(db_error)
                            
                            # Tạo file Excel với kết quả đánh giá chi tiết
                            try:
                                logger.info(f"📊 Đang tạo file Excel cho kết quả đánh giá...")
                                excel_path = export_dance_scoring_to_excel(result_data, user_request_id)
                                result_data['excel_download_url'] = f"/res/output/{os.path.basename(excel_path)}"
                                logger.info(f"✅ Đã tạo file Excel thành công: {excel_path}")
                            except Exception as excel_error:
                                logger.error(f"❌ Lỗi tạo file Excel: {str(excel_error)}")
                                # Không dừng xử lý nếu tạo Excel thất bại

                            # 📊 TỰ ĐỘNG GHI VÀO GOOGLE SHEETS 📊
                            if sheets_helper:
                                try:
                                    logger.info(f"📊 Đang ghi kết quả vào Google Sheets...")

                                    # Đảm bảo có thể truy cập spreadsheet, tạo mới nếu cần
                                    if not sheets_helper.ensure_spreadsheet_access():
                                        logger.error("❌ Không thể truy cập hoặc tạo Google Sheets")
                                        raise Exception("Google Sheets access failed")

                                    # Ghi kết quả vào các sheet mặc định
                                    sheets_helper.write_complete_result(result_data)

                                    # Ghi vào sheet "Mô phỏng" với 5 trường đặc biệt và logic random
                                    try:
                                        mophong_scores = sheets_helper.write_mophong_data(result_data, 'Mophong')
                                        result_data['mophong_scores'] = mophong_scores
                                        logger.info(f"✅ Đã ghi thành công vào sheet 'Mophong' với dữ liệu mô phỏng!")
                                    except Exception as custom_error:
                                        logger.error(f"❌ Lỗi ghi vào sheet 'Mophong': {str(custom_error)}")
                                        # Không dừng xử lý nếu ghi sheet tùy chỉnh thất bại

                                    logger.info(f"✅ Đã ghi thành công vào Google Sheets!")

                                except Exception as sheets_error:
                                    logger.error(f"❌ Lỗi ghi vào Google Sheets: {str(sheets_error)}")
                                    logger.error(f"🔍 Chi tiết lỗi: {traceback.format_exc()}")
                                    # Không dừng xử lý nếu ghi Google Sheets thất bại

                            yield {"event": "result", "data": json.dumps(result_data)}
                            
                            # Thêm một type result nữa để thông báo hoàn thành và tóm tắt
                            yield {"event": "result", "data": json.dumps({
                                "type": "completed",
                                "message": "Video comparison completed successfully",
                                "result_url": result_data.get('cloudinary_video_url', result_data.get('result_url')),
                                "excel_url": result_data.get('excel_download_url'),
                                "average_score": result_data.get('average_similarity_score'),
                                "dance_metrics": result_data.get('dance_scoring_metrics', {}),
                                "google_sheets_status": "success" if sheets_helper else "disabled"
                            })}
                        elif message["type"] == "error":
                            error_data_msg = message["data"]
                            
                            # Save error to database if requested
                            if save_to_db:
                                try:
                                    logger.info(f"💾 Saving error to database for user: {user_id}")
                                    db_error_data = {
                                        'error': error_data_msg,
                                        'input_video_url': user_video.filename,
                                        'title': title or 'Video Comparison Failed',
                                        'user_id': user_id
                                    }
                                    
                                    db_manager.save_video_result(
                                        db_error_data,
                                        'compare_videos',
                                        user_id
                                    )
                                    logger.info(f"✅ Successfully saved error to database")
                                    
                                except Exception as db_error:
                                    logger.error(f"❌ Failed to save error to database: {str(db_error)}")
                                    logger.error(f"🔍 DB Error Details: {traceback.format_exc()}")
                            
                            # Ghi lỗi vào Google Sheets nếu có thể
                            if sheets_helper:
                                try:
                                    # Đảm bảo có thể truy cập spreadsheet, tạo mới nếu cần
                                    if sheets_helper.ensure_spreadsheet_access():
                                        # Ghi lỗi vào sheet tổng quan
                                        error_result = {
                                            'error': error_data_msg,
                                            'input_video_url': user_video.filename,
                                            'user_id': user_id,
                                            'title': title or 'Video Comparison Failed',
                                            'timestamp': datetime.now().isoformat()
                                        }
                                        sheets_helper.write_overview_data(error_result, 'Tongquan')
                                        logger.info(f"✅ Đã ghi lỗi vào Google Sheets")

                                        # Ghi lỗi vào sheet "Mô phỏng" với định dạng đơn giản
                                        try:
                                            error_columns = ['thời gian', 'trạng thái', 'thông báo lỗi']
                                            error_data_custom = {
                                                'timestamp': datetime.now().isoformat(),
                                                'status': 'error',
                                                'error_message': error_data_msg
                                            }
                                            sheets_helper.write_custom_sheet(error_data_custom, 'Mophong', error_columns)
                                            logger.info(f"✅ Đã ghi lỗi vào sheet 'Mophong'")
                                        except Exception as custom_error:
                                            logger.error(f"❌ Lỗi ghi lỗi vào sheet 'Mophong': {str(custom_error)}")
                                except Exception as sheets_error:
                                    logger.error(f"❌ Lỗi ghi lỗi vào Google Sheets: {str(sheets_error)}")
                            
                            yield {"event": "error", "data": error_data_msg}
                            break
                        elif message["type"] == "progress":
                            # Send detailed progress updates
                            yield {"event": "progress", "data": json.dumps(message)}
                    else:
                        yield {"event": "log", "data": message}
                except queue.Empty:
                    if not comparison_thread.is_alive():
                        break
                    await asyncio.sleep(0.1)
        finally:
            comparison_thread.join()
            cleanup_files(files_to_cleanup)
    return EventSourceResponse(event_generator())


def run_video_annotation_in_thread(user_video_path, reference_video_path, output_path, result_queue):
    """
    Runs the video annotation process in a thread.
    Compares user video to reference and creates an annotated video with feedback.
    """
    try:
        logger.info(f"🔄 Starting video annotation process...")
        logger.info(f"   📹 User video: {user_video_path}")
        logger.info(f"   🎬 Reference video: {reference_video_path}")
        logger.info(f"   💾 Output: {output_path}")

        # Initialize pose comparison
        logger.info(f"🤖 Initializing PoseComparison with reference video...")
        comparison = PoseComparison(reference_video_path)

        # Process annotation
        logger.info(f"🎨 Starting video annotation process...")
        comparison.annotate_video(user_video_path, output_path)

        logger.info(f"✅ Video annotation completed successfully!")
        logger.info(f"   📊 Output saved to: {output_path}")
        result_queue.put({"type": "result", "data": {"output_path": output_path}})

    except Exception as e:
        error_msg = f"❌ Error in video annotation: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"🔍 Debug info - User video: {user_video_path}, Ref video: {reference_video_path}")
        logger.debug(f"📋 Full traceback: {traceback.format_exc()}")
        result_queue.put({"type": "error", "data": error_msg})
    finally:
        logger.info(f"🏁 Video annotation thread finished")
        result_queue.put({"type": "done"})


@app.post("/api/analyze_performance/", 
          summary="Analyze user video and provide feedback video",
          tags=["Video Comparison"])
async def analyze_performance(
    user_video: UploadFile = File(...), 
    reference_video_id: str = Form(None),
    reference_video: UploadFile = File(None)
):
    """
    Upload a user's performance video to compare against a reference video.

    You can either provide the ID of a pre-existing reference video via `reference_video_id`
    or upload a new reference video directly via `reference_video`.

    This endpoint processes the video in the background and returns an annotated 
    video showing the user's pose with color-coded feedback on correctness 
    compared to the reference pose.

    The process is streamed using Server-Sent Events (SSE).
    """
    
    if not reference_video and not reference_video_id:
        raise HTTPException(status_code=400, detail="You must provide either a 'reference_video_id' or upload a 'reference_video'.")

    user_video_path, _ = await save_upload_file(user_video)
    
    ref_video_path = None
    files_to_cleanup = [user_video_path]

    try:
        if reference_video:
            # User uploaded a new reference video, save it temporarily
            ref_video_path, _ = await save_upload_file(reference_video)
            files_to_cleanup.append(ref_video_path)
        elif reference_video_id:
            # User selected an existing reference video
            ref_video_path = os.path.join(REFERENCE_VIDEOS_DIR, reference_video_id)
            if not os.path.isfile(ref_video_path):
                raise HTTPException(status_code=404, detail=f"Reference video not found: {reference_video_id}")

        user_video_basename = os.path.basename(user_video_path)
        annotated_filename = f"annotated_{user_video_basename}"
        output_path = os.path.join(OUTPUTS_DIR, annotated_filename)

        async def event_generator():
            result_queue = queue.Queue()
            
            annotation_thread = threading.Thread(
                target=run_video_annotation_in_thread,
                args=(user_video_path, ref_video_path, output_path, result_queue)
            )
            annotation_thread.start()
            
            try:
                while True:
                    try:
                        message = result_queue.get_nowait()
                        if isinstance(message, dict):
                            event_type = message.get("type", "log")
                            data = message.get("data", "")
                            
                            if event_type == "done":
                                yield {"event": "done", "data": "Processing finished."}
                                break
                            elif event_type == "result":
                                yield {"event": "result", "data": json.dumps(data)}
                            elif event_type == "error":
                                yield {"event": "error", "data": data}
                                break
                            else: # log
                                yield {"event": "log", "data": data}
                    except queue.Empty:
                        if not annotation_thread.is_alive():
                            break
                        await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in event generator: {e}")
                yield {"event": "error", "data": str(e)}
            finally:
                annotation_thread.join()
                cleanup_files(files_to_cleanup)
        
        return EventSourceResponse(event_generator())

    except HTTPException as e:
        # If an HTTPException was raised, ensure cleanup is performed
        cleanup_files(files_to_cleanup)
        raise e

@app.websocket("/ws/compare_live/{reference_video_id}")
async def websocket_compare_live(websocket: WebSocket, reference_video_id: str):
    """
    Handles a live comparison session via WebSocket.
    - It streams webcam frames to the server.
    - The server streams back real-time comparison results (score, keypoints).
    - After disconnection, a final annotated video of the performance is saved.
    """
    await websocket.accept()
    session: LiveComparisonSession = None

    # Construct the path and check if the reference video exists
    ref_video_path = os.path.join(REFERENCE_VIDEOS_DIR, reference_video_id)
    if not os.path.isfile(ref_video_path):
        await websocket.send_json({"type": "error", "message": f"Reference video not found: {reference_video_id}"})
        await websocket.close(code=1008, reason="Reference video not found")
        return

    try:
        # Initialize the session
        session = LiveComparisonSession(ref_video_path, OUTPUTS_DIR)
        await websocket.send_json({"type": "session_started", "output_video_path": session.output_path})

        # Loop to process frames from the client
        while True:
            user_frame_bytes = await websocket.receive_bytes()
            result = session.process_frame(user_frame_bytes)
            await websocket.send_json({"type": "comparison_result", **result})

    except WebSocketDisconnect:
        logger.info("Client disconnected from live session.")
    except Exception as e:
        logger.error(f"An error occurred during live session: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        logger.info(f"Error details - Reference video ID: {reference_video_id}")
    finally:
        # Clean up the session and post-process the recorded video
        if session:
            logger.info("Session closed. Starting post-processing of recorded video...")
            raw_video_path = session.close()

            annotated_filename = os.path.basename(raw_video_path).replace("live_session_", "annotated_")
            annotated_video_path = os.path.join(OUTPUTS_DIR, annotated_filename)

            try:
                logger.info(f"Annotating video: {raw_video_path} -> {annotated_video_path}")
                # Re-create a comparison object for annotation as the session one is closed
                annotation_comparison = PoseComparison(ref_video_path)
                annotation_comparison.annotate_video(raw_video_path, annotated_video_path)
                
                logger.info(f"Cleaning up raw file: {raw_video_path}")
                os.remove(raw_video_path)
                logger.info(f"✅ Final annotated video is ready at: {annotated_video_path}")

            except Exception as post_process_error:
                logger.error(f"Error during video post-processing: {post_process_error}")


@app.websocket("/ws/live_camera_analysis/{reference_video_id}")
async def websocket_live_camera_analysis(websocket: WebSocket, reference_video_id: str):
    """
    WebSocket endpoint for live camera analysis with pose detection.

    Features:
    - Real-time pose detection from camera frames
    - Session recording and storage
    - Post-session analysis against reference video
    - Live feedback with pose keypoints and confidence scores

    Protocol:
    1. Client connects and sends 'start' message
    2. Client sends camera frames as binary data
    3. Server processes frames and sends back pose analysis
    4. Client sends 'stop' message to end session
    5. Server performs analysis and returns results
    """
    await websocket.accept()
    session = None

    # Construct the path and check if the reference video exists
    ref_video_path = os.path.join(REFERENCE_VIDEOS_DIR, reference_video_id)
    if not os.path.isfile(ref_video_path):
        await websocket.send_json({
            "type": "error",
            "message": f"Reference video not found: {reference_video_id}"
        })
        await websocket.close(code=1008, reason="Reference video not found")
        return

    try:
        # Initialize the camera session
        logger.info(f"🎥 Initializing LiveCameraSession for reference video: {reference_video_id}")
        session = LiveCameraSession(OUTPUTS_DIR, ref_video_path)
        logger.info(f"✅ Camera session initialized - Session ID: {session.session_id}")

        await websocket.send_json({
            "type": "session_initialized",
            "session_id": session.session_id,
            "message": f"Live camera session ready - ID: {session.session_id}"
        })
        logger.info("📤 Sent session_initialized message to client")

        while True:
            # Receive message from client
            logger.debug("🔄 Waiting for message from client...")
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                logger.info("📴 Client disconnected from camera session")
                break

            # Handle text messages (start/stop commands)
            if message["type"] == "websocket.text":
                data = json.loads(message["text"])
                logger.info(f"📨 Received text message: {data}")

                if data.get("action") == "start":
                    logger.info("▶️ Starting camera session")
                    session.start_session()
                    logger.info("✅ Camera session started successfully")

                    await websocket.send_json({
                        "type": "session_started",
                        "message": "Camera session started - send frames as binary data"
                    })
                    logger.info("📤 Sent session_started message to client")

                elif data.get("action") == "stop":
                    logger.info("⏹️ Stopping camera session")
                    session_info = session.stop_session()
                    logger.info(f"✅ Camera session stopped - {session_info}")

                    await websocket.send_json({
                        "type": "session_stopped",
                        "session_info": session_info,
                        "message": "Session stopped - analyzing..."
                    })
                    logger.info("📤 Sent session_stopped message to client")

                    # Perform analysis
                    logger.info("🔍 Starting session analysis...")
                    analysis_result = session.analyze_session(ref_video_path)
                    logger.info(f"✅ Session analysis completed: {analysis_result}")

                    await websocket.send_json({
                        "type": "analysis_complete",
                        "result": analysis_result,
                        "message": "Analysis complete!"
                    })
                    logger.info("📤 Sent analysis_complete message to client")
                    break

            # Handle binary data (camera frames)
            elif message["type"] == "websocket.bytes":
                if session and session.is_active:
                    frame_bytes = message["bytes"]
                    logger.debug(f"📸 Received frame - Size: {len(frame_bytes)} bytes")

                    # Process frame
                    logger.debug("⚙️ Processing camera frame...")
                    result = session.process_frame(frame_bytes)
                    logger.debug(f"✅ Frame processed - Result: {result}")

                    if "error" not in result:
                        # Send real-time feedback
                        await websocket.send_json({
                            "type": "frame_processed",
                            "frame_info": result,
                            "session_stats": {
                                "total_frames": len(session.frames),
                                "total_pose_data": len(session.pose_data),
                                "duration": result.get("timestamp", 0)
                            }
                        })
                        logger.debug("📤 Sent frame_processed message to client")
                    else:
                        logger.warning(f"⚠️ Frame processing error: {result['error']}")
                        await websocket.send_json({
                            "type": "frame_error",
                            "error": result["error"]
                        })
                        logger.debug("📤 Sent frame_error message to client")

    except WebSocketDisconnect:
        logger.info("📴 Client disconnected from camera session")
    except Exception as e:
        logger.error(f"❌ Error in live camera session: {e}")
        logger.debug(f"📋 Full traceback: {traceback.format_exc()}")
        await websocket.send_json({
            "type": "error",
            "message": f"Server error: {str(e)}"
        })
        logger.error("📤 Sent error message to client")
    finally:
        # Clean up session
        if session:
            try:
                logger.info("🧹 Starting camera session cleanup...")
                session_info = session.stop_session()
                logger.info(f"✅ Camera session cleanup completed: {session_info}")
            except Exception as e:
                logger.error(f"❌ Error during session cleanup: {e}")
                logger.debug(f"📋 Cleanup error traceback: {traceback.format_exc()}")


# --- Database Integration APIs ---

@app.post("/api/process_and_save_compare_videos/",
          summary="Compare videos and save result to database",
          tags=["Database Integration"])
async def process_and_save_compare_videos(
    user_video: UploadFile = File(...),
    reference_video: UploadFile = File(...),
    user_id: str = Form(None),
    title: str = Form(None)
):
    """
    Compare two videos and automatically save the result to database.
    Returns both the processing result and the database record ID.
    """
    try:
        # Create a modified version of compare_videos that saves to DB
        user_video_path, user_request_id = await save_upload_file(user_video)
        ref_video_path, _ = await save_upload_file(reference_video)
        
        output_filename = f"comparison_{user_request_id}.mp4"
        output_path = os.path.join(OUTPUTS_DIR, output_filename)

        async def event_generator():
            result_queue = queue.Queue()
            files_to_cleanup = [user_video_path, ref_video_path]
            
            comparison_thread = threading.Thread(
                target=run_video_comparison_in_thread,
                args=(user_video_path, ref_video_path, output_path, result_queue)
            )
            comparison_thread.start()

            try:
                while True:
                    try:
                        message = result_queue.get_nowait()
                        if isinstance(message, dict):
                            if message["type"] == "done":
                                yield {"event": "done", "data": "Processing finished."}
                                break
                            elif message["type"] == "result":
                                # Save result to database
                                logger.info(f"💾 Bắt đầu lưu kết quả vào database cho user: {user_id}")
                                result_data = message["data"]
                                
                                # Add metadata
                                result_data['input_video_url'] = user_video.filename
                                if user_id:
                                    result_data['user_id'] = user_id
                                if title:
                                    result_data['title'] = title
                                
                                # Use cloudinary_video_url as result_url for database storage
                                if 'cloudinary_video_url' in result_data:
                                    result_data['result_url'] = result_data['cloudinary_video_url']
                                
                                # Save to database
                                logger.info(f"📤 Gửi dữ liệu đến database manager...")
                                record_id = db_manager.save_video_result(
                                    result_data, 
                                    'compare_videos',
                                    user_id
                                )
                                
                                # Add database ID to result
                                result_data['database_record_id'] = record_id
                                logger.info(f"✅ Thành công! Đã lưu vào database với ID: {record_id}")
                                yield {"event": "result", "data": json.dumps(result_data)}
                                
                            elif message["type"] == "error":
                                # Save error to database
                                logger.info(f"💾 Bắt đầu lưu lỗi vào database cho user: {user_id}")
                                error_data = {
                                    'error': message['data'],
                                    'input_video_url': user_video.filename,
                                    'title': title or 'Video Comparison Failed'
                                }
                                
                                if user_id:
                                    error_data['user_id'] = user_id
                                
                                logger.info(f"📤 Gửi dữ liệu lỗi đến database manager...")
                                db_manager.save_video_result(
                                    error_data,
                                    'compare_videos',
                                    user_id
                                )
                                logger.info(f"✅ Thành công! Đã lưu lỗi vào database")
                                yield {"event": "error", "data": message["data"]}
                                break
                            elif message["type"] == "progress":
                                yield {"event": "progress", "data": json.dumps(message)}
                    except queue.Empty:
                        if not comparison_thread.is_alive():
                            break
                        await asyncio.sleep(0.1)
            finally:
                comparison_thread.join()
                cleanup_files(files_to_cleanup)
        
        return EventSourceResponse(event_generator())
        
    except Exception as e:
        logger.error(f"❌ Error in process_and_save_compare_videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process_and_save_video_stream/",
          summary="Process video stream and save result to database",
          tags=["Database Integration"])
async def process_and_save_video_stream(
    file: UploadFile = File(...),
    user_id: str = Form(None),
    title: str = Form(None)
):
    """
    Process video for 3D pose estimation and save result to database.
    """
    try:
        video_path, request_id = await save_upload_file(file)
        
        async def event_generator():
            result_queue = queue.Queue()
            files_to_cleanup = [video_path]
            
            pipeline_thread = threading.Thread(
                target=run_pipeline_in_thread,
                args=(video_path, OUTPUTS_DIR, request_id, result_queue)
            )
            pipeline_thread.start()
            
            try:
                while True:
                    try:
                        message = result_queue.get_nowait()
                        if isinstance(message, dict):
                            if message["type"] == "done":
                                yield {"event": "done", "data": "Processing finished."}
                                break
                            elif message["type"] == "result":
                                # Process and save result
                                final_json_path, generated_files = message["data"]
                                files_to_cleanup.extend(generated_files)
                                
                                # Read result data
                                with open(final_json_path, 'r') as f:
                                    result_data = json.load(f)
                                
                                # Add metadata
                                result_data['input_video_url'] = file.filename
                                result_data['result_url'] = f"/res/output/{os.path.basename(final_json_path)}"
                                if user_id:
                                    result_data['user_id'] = user_id
                                if title:
                                    result_data['title'] = title
                                
                                # Use cloudinary_video_url as result_url for database storage
                                if 'cloudinary_video_url' in result_data:
                                    result_data['result_url'] = result_data['cloudinary_video_url']
                                
                                # Save to database
                                logger.info(f"💾 Bắt đầu lưu kết quả vào database cho user: {user_id}")
                                record_id = db_manager.save_video_result(
                                    result_data,
                                    'process_video_stream',
                                    user_id
                                )
                                
                                # Add database ID
                                result_data['database_record_id'] = record_id
                                logger.info(f"✅ Thành công! Đã lưu vào database với ID: {record_id}")
                                yield {"event": "result", "data": json.dumps(result_data)}
                                
                            elif message["type"] == "error":
                                # Save error to database
                                logger.info(f"💾 Bắt đầu lưu lỗi vào database cho user: {user_id}")
                                error_data = {
                                    'error': message['data'],
                                    'input_video_url': file.filename,
                                    'title': title or 'Video Processing Failed'
                                }
                                
                                if user_id:
                                    error_data['user_id'] = user_id
                                
                                logger.info(f"📤 Gửi dữ liệu lỗi đến database manager...")
                                db_manager.save_video_result(
                                    error_data,
                                    'process_video_stream',
                                    user_id
                                )
                                logger.info(f"✅ Thành công! Đã lưu lỗi vào database")
                                yield {"event": "error", "data": message["data"]}
                                break
                            else:
                                yield {"event": "log", "data": message}
                    except queue.Empty:
                        if not pipeline_thread.is_alive():
                            break
                        await asyncio.sleep(0.1)
            finally:
                pipeline_thread.join()
                cleanup_files(files_to_cleanup)
        
        return EventSourceResponse(event_generator())
        
    except Exception as e:
        logger.error(f"❌ Error in process_and_save_video_stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get_compare_videos_results/",
         summary="Get compare videos results from database",
         tags=["Database Integration"])
async def get_compare_videos_results(
    user_id: str = None,
    limit: int = 50,
    offset: int = 0
):
    """
    Retrieve compare videos results from database.
    """
    try:
        results = db_manager.get_video_results(
            process_type='compare_videos',
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        return {
            "success": True,
            "count": len(results),
            "data": results
        }
    except Exception as e:
        logger.error(f"❌ Error retrieving compare videos results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get_video_stream_results/",
         summary="Get video stream processing results from database",
         tags=["Database Integration"])
async def get_video_stream_results(
    user_id: str = None,
    limit: int = 50,
    offset: int = 0
):
    """
    Retrieve video stream processing results from database.
    """
    try:
        results = db_manager.get_video_results(
            process_type='process_video_stream',
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        return {
            "success": True,
            "count": len(results),
            "data": results
        }
    except Exception as e:
        logger.error(f"❌ Error retrieving video stream results: {e}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    logger.info("🚀 Starting FastAPI server v2.0...")
    logger.info(f"Access http://{host}:{port}/docs for the interactive API documentation.")
    uvicorn.run(app, host=host, port=port)