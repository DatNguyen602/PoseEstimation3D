import os
import logging
from typing import Optional, Dict, Any
import cloudinary
from cloudinary import uploader
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

logger = logging.getLogger(__name__)

class CloudinaryHelper:
    """Helper class for Cloudinary operations"""

    @staticmethod
    def upload_video(file_path: str, public_id: Optional[str] = None,
                    folder: str = None, **kwargs) -> Dict[str, Any]:
        """
        Upload video to Cloudinary

        Args:
            file_path: Local path to video file
            public_id: Custom public ID for the video (optional)
            folder: Folder in Cloudinary to upload to (optional)
            **kwargs: Additional upload options

        Returns:
            Dict containing upload result with 'url', 'public_id', etc.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Video file not found: {file_path}")

            # Set default folder if not provided
            if folder is None:
                folder = os.getenv('CLOUDINARY_UPLOAD_FOLDER', 'pose_estimation_videos')

            # Prepare upload options
            upload_options = {
                'resource_type': 'video',
                'folder': folder,
                'format': 'mp4',        # THÊM DÒNG NÀY → BẮT BUỘC CHUYỂN THÀNH MP4
                **kwargs
            }

            if public_id:
                upload_options['public_id'] = public_id

            logger.info(f"📤 Uploading video to Cloudinary: {file_path}")

            # Upload video
            result = uploader.upload(file_path, **upload_options)

            logger.info(f"✅ Video uploaded successfully: {result['public_id']}")

            return {
                'success': True,
                'url': result['url'],
                'secure_url': result['secure_url'],
                'public_id': result['public_id'],
                'format': result.get('format'),
                'width': result.get('width'),
                'height': result.get('height'),
                'duration': result.get('duration'),
                'size': result.get('bytes')
            }

        except Exception as e:
            logger.error(f"❌ Failed to upload video to Cloudinary: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    @staticmethod
    def get_video_url(public_id: str, transformations: Optional[Dict] = None) -> str:
        """
        Get optimized video URL from Cloudinary

        Args:
            public_id: Cloudinary public ID of the video
            transformations: Video transformation options

        Returns:
            Optimized video URL
        """
        try:
            # Default transformations for video streaming
            default_transforms = {
                'quality': 'auto',
                'format': 'auto',
                'controls': True
            }

            if transformations:
                default_transforms.update(transformations)

            # Create URL with transformations
            url, _ = cloudinary.utils.cloudinary_url(
                public_id,
                resource_type='video',
                **default_transforms
            )

            return url

        except Exception as e:
            logger.error(f"❌ Failed to generate video URL: {str(e)}")
            return None

    @staticmethod
    def get_adaptive_streaming_url(public_id: str) -> str:
        """
        Get adaptive bitrate streaming URL (HLS)

        Args:
            public_id: Cloudinary public ID of the video

        Returns:
            HLS streaming URL
        """
        try:
            # Create HLS streaming URL
            url, _ = cloudinary.utils.cloudinary_url(
                public_id,
                resource_type='video',
                format='m3u8',
                streaming_profile='auto'
            )

            return url

        except Exception as e:
            logger.error(f"❌ Failed to generate streaming URL: {str(e)}")
            return None

    @staticmethod
    def delete_video(public_id: str) -> bool:
        """
        Delete video from Cloudinary

        Args:
            public_id: Cloudinary public ID of the video

        Returns:
            True if successful, False otherwise
        """
        try:
            result = uploader.destroy(public_id, resource_type='video')
            if result['result'] == 'ok':
                logger.info(f"✅ Video deleted from Cloudinary: {public_id}")
                return True
            else:
                logger.error(f"❌ Failed to delete video: {result}")
                return False

        except Exception as e:
            logger.error(f"❌ Error deleting video from Cloudinary: {str(e)}")
            return False

# Convenience functions
def upload_comparison_video(file_path: str, request_id: str) -> Dict[str, Any]:
    """
    Upload comparison video with standard naming convention

    Args:
        file_path: Local path to video file
        request_id: Request ID for naming

    Returns:
        Upload result dict
    """
    public_id = f"comparison_video_{request_id}"
    return CloudinaryHelper.upload_video(
        file_path=file_path,
        public_id=public_id,
        folder='comparison_videos'
    )

def upload_annotated_video(file_path: str, request_id: str) -> Dict[str, Any]:
    """
    Upload annotated video with standard naming convention

    Args:
        file_path: Local path to video file
        request_id: Request ID for naming

    Returns:
        Upload result dict
    """
    public_id = f"annotated_video_{request_id}"
    return CloudinaryHelper.upload_video(
        file_path=file_path,
        public_id=public_id,
        folder='annotated_videos'
    )
