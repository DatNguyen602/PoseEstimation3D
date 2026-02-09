from pydantic import BaseModel
from typing import List

class CompareRequest(BaseModel):
    """
    Model for the pose comparison request.
    """
    user_image: str  # Base64 encoded string of the user's image
    reference_video_path: str
    reference_frame_index: int

class CompareResponse(BaseModel):
    """
    Model for the pose comparison response.
    """
    score: float
    wrong_keypoints: List[int]
    total_keypoints: int
