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
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from pose_comparison import PoseComparison, LiveComparisonSession
from api_models import CompareRequest, CompareResponse

# Import the pipeline function
from run_pipeline import run_full_pipeline

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

# Mount static files directory for reference videos
app.mount("/reference_videos", StaticFiles(directory=REFERENCE_VIDEOS_DIR), name="reference_videos")

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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

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
    print(f"🧹 Starting cleanup for {len(files)} files...")
    for file_path in files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ Deleted: {file_path}")
        except Exception as e:
            print(f"⚠️ Error cleaning up file {file_path}: {e}")

async def save_upload_file(upload_file: UploadFile) -> str:
    """Saves a temporary uploaded file and returns its path and request ID."""
    request_id = str(uuid.uuid4())
    _, extension = os.path.splitext(upload_file.filename)
    if extension.lower() not in ['.mp4', '.mov', '.avi']:
        raise HTTPException(status_code=400, detail="Invalid file format.")
    
    video_filename = f"{request_id}{extension}"
    video_path = os.path.join(UPLOADS_DIR, video_filename)

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()
    
    print(f"📹 Saved temporary video to: {video_path}")
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
    
    print(f"📹 Saved new reference video: {video_path}")
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
    try:
        original_stdout = sys.stdout
        sys.stdout = QueueIO(result_queue)
        result = run_full_pipeline(video_path, output_dir, output_basename)
        result_queue.put({"type": "result", "data": result})
    except Exception as e:
        error_str = traceback.format_exc()
        result_queue.put({"type": "error", "data": error_str})
    finally:
        sys.stdout = original_stdout
        result_queue.put({"type": "done"})

@app.post("/process-video-stream/", 
          summary="[Legacy] Upload video for 3D pose estimation (SSE)",
          tags=["Legacy"])
async def process_video_stream(file: UploadFile = File(...)):
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
                            final_json_path, generated_files = message["data"]
                            files_to_cleanup.extend(generated_files)
                            with open(final_json_path, 'r') as f:
                                json_data = json.load(f)
                            yield {"event": "result", "data": json.dumps(json_data)}
                        elif message["type"] == "error":
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

def run_video_comparison_in_thread(user_video_path, reference_video_path, output_path, result_queue):
    try:
        result_queue.put("Starting side-by-side comparison...")
        comparison_sbs = PoseComparison(reference_video_path)
        comparison_sbs.process_video_files(user_video_path, output_path, result_queue)
        result_queue.put(f"Side-by-side video created: {output_path}")

        result_queue.put("Starting user video annotation...")
        output_dir = os.path.dirname(output_path)
        user_video_basename = os.path.basename(user_video_path)
        annotated_filename = f"annotated_{user_video_basename}"
        annotated_output_path = os.path.join(output_dir, annotated_filename)

        comparison_anno = PoseComparison(reference_video_path)
        comparison_anno.annotate_video(user_video_path, annotated_output_path)
        result_queue.put(f"Annotated video created: {annotated_output_path}")

        result_data = {
            "side_by_side_video_path": output_path,
            "annotated_user_video_path": annotated_output_path
        }
        result_queue.put({"type": "result", "data": result_data})
    except Exception as e:
        error_str = traceback.format_exc()
        result_queue.put({"type": "error", "data": error_str})
    finally:
        result_queue.put({"type": "done"})

@app.post("/api/compare_videos/", 
          summary="Compare two uploaded videos (batch processing)",
          tags=["Video Comparison"])
async def compare_videos(user_video: UploadFile = File(...), reference_video: UploadFile = File(...)):
    user_video_path, user_request_id = await save_upload_file(user_video)
    ref_video_path, _ = await save_upload_file(reference_video)
    
    output_filename = f"comparison_{user_request_id}.mp4"
    output_path = os.path.join(OUTPUTS_DIR, output_filename)

    user_video_basename = os.path.basename(user_video_path)
    annotated_filename = f"annotated_{user_video_basename}"
    annotated_output_path = os.path.join(OUTPUTS_DIR, annotated_filename)

    async def event_generator():
        result_queue = queue.Queue()
        files_to_cleanup = [user_video_path, ref_video_path, output_path, annotated_output_path]
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
                            yield {"event": "result", "data": json.dumps(message["data"])}
                        elif message["type"] == "error":
                            yield {"event": "error", "data": message["data"]}
                            break
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

@app.websocket("/ws/compare_live/{reference_video_id}")
async def websocket_compare_live(websocket: WebSocket, reference_video_id: str):
    """
    Handles a live comparison session via WebSocket.
    - The client connects to this endpoint with a `reference_video_id`.
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
        print("Client disconnected from live session.")
    except Exception as e:
        print(f"An error occurred during live session: {e}")
        traceback.print_exc()
    finally:
        # Clean up the session and post-process the recorded video
        if session:
            print("Session closed. Starting post-processing of recorded video...")
            raw_video_path = session.close()

            annotated_filename = os.path.basename(raw_video_path).replace("live_session_", "annotated_")
            annotated_video_path = os.path.join(OUTPUTS_DIR, annotated_filename)

            try:
                print(f"Annotating video: {raw_video_path} -> {annotated_video_path}")
                # Re-create a comparison object for annotation as the session one is closed
                annotation_comparison = PoseComparison(ref_video_path)
                annotation_comparison.annotate_video(raw_video_path, annotated_video_path)
                
                print(f"Cleaning up raw file: {raw_video_path}")
                os.remove(raw_video_path)
                print(f"✅ Final annotated video is ready at: {annotated_video_path}")

            except Exception as post_process_error:
                print(f"Error during video post-processing: {post_process_error}")


@app.get("/", summary="API Root", include_in_schema=False)
def read_root():
    return {"message": "Welcome to the Pose Comparison API v2.0. See /docs for details."}


if __name__ == "__main__":
    print("🚀 Starting FastAPI server v2.0...")
    print("Access http://127.0.0.1:8000/docs for the interactive API documentation.")
    uvicorn.run(app, host="127.0.0.1", port=8000)
