import torch
import numpy as np
import sys
import os
from pathlib import Path
import warnings

# Setup VideoPose3D path
def setup_videopose3d_path():
    """Setup path cho VideoPose3D repository"""
    possible_paths = [
        Path("VideoPose3D"),
        Path("../VideoPose3D"),
        Path("./VideoPose3D"),
        Path(os.environ.get("VIDEOPOSE3D_PATH", "VideoPose3D"))
    ]
    
    for videopose3d_path in possible_paths:
        if videopose3d_path.exists() and (videopose3d_path / "common").exists():
            abs_path = str(videopose3d_path.resolve())
            if abs_path not in sys.path:
                sys.path.insert(0, abs_path)
            print(f"✅ Found VideoPose3D at: {videopose3d_path}")
            return True, str(videopose3d_path)
    
    print(f"❌ VideoPose3D repository not found!")
    print(f"Searched in: {[str(p) for p in possible_paths]}")
    print(f"Please ensure VideoPose3D is cloned in one of these locations:")
    for p in possible_paths:
        print(f"  - {p}")
    return False, None

# Setup VideoPose3D
VIDEOPOSE3D_AVAILABLE, VIDEOPOSE3D_PATH = setup_videopose3d_path()

# Import VideoPose3D modules
if VIDEOPOSE3D_AVAILABLE:
    try:
        from VideoPose3D.common.model import TemporalModel
        from VideoPose3D.common.camera import normalize_screen_coordinates, camera_to_world
        from VideoPose3D.common.loss import mpjpe, p_mpjpe
        from VideoPose3D.common.utils import deterministic_random
        
        print("✅ Successfully imported VideoPose3D modules")
        print(f"   - TemporalModel: Available")
        print(f"   - Camera utilities: Available") 
        print(f"   - Loss functions: Available")
        
    except ImportError as e:
        print(f"⚠️ Import error from VideoPose3D: {e}")
        print(f"Please check VideoPose3D installation:")
        print(f"1. Ensure repository is properly cloned")
        print(f"2. Check common/ directory exists")
        print(f"3. Verify Python path setup")
        VIDEOPOSE3D_AVAILABLE = False

class VideoPose3DPredictor:
    """
    High-level interface for VideoPose3D model với official repository
    """
    
    def __init__(self, checkpoint_path=None, device=None, architecture=None):
        """
        Initialize VideoPose3D predictor
        
        Args:
            checkpoint_path: Path to pretrained checkpoint (.bin file)
            device: PyTorch device (auto-detect if None)
            architecture: Model architecture (default: [3,3,3,3,3])
        """
        if not VIDEOPOSE3D_AVAILABLE:
            raise RuntimeError("VideoPose3D repository not found! Please clone the official repository.")
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.architecture = architecture or [3, 3, 3, 3, 3]
        
        print(f"🤖 Initializing VideoPose3D Predictor...")
        print(f"   Device: {self.device}")
        print(f"   Architecture: {self.architecture}")
        
        # Initialize model
        self.model = self._create_model()
        
        # Load checkpoint
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        else:
            checkpoint_path = self._find_default_checkpoint()
            if checkpoint_path:
                self._load_checkpoint(checkpoint_path)
            else:
                print("⚠️ No checkpoint found. Using random weights (poor accuracy expected)")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Model properties
        self.receptive_field = self.model.receptive_field()
        print(f"   Receptive field: {self.receptive_field} frames")
        print(f"✅ VideoPose3D predictor ready!")
    
    def _find_default_checkpoint(self):
        """Tìm default checkpoint trong VideoPose3D directory"""
        if not VIDEOPOSE3D_PATH:
            return None
        
        checkpoint_paths = [
            Path(VIDEOPOSE3D_PATH) / "checkpoint" / "pretrained_h36m_detectron_coco.bin",
            Path(VIDEOPOSE3D_PATH) / "pretrained_h36m_detectron_coco.bin",
            Path("checkpoint") / "pretrained_h36m_detectron_coco.bin",
            Path("pretrained_h36m_detectron_coco.bin")
        ]
        
        for checkpoint_path in checkpoint_paths:
            if checkpoint_path.exists():
                print(f"🔍 Found default checkpoint: {checkpoint_path}")
                return str(checkpoint_path)
        
        print(f"❌ No default checkpoint found. Searched:")
        for p in checkpoint_paths:
            print(f"   - {p}")
        return None
    
    def _create_model(self):
        """Create TemporalModel với specified architecture"""
        try:
            model = TemporalModel(
                num_joints_in=17,       # COCO format
                in_features=2,          # x, y coordinates
                num_joints_out=17,      # Human3.6M format
                filter_widths=self.architecture,
                causal=False,           # Non-causal (uses future frames)
                dropout=0.25,
                channels=1024,
                dense=False
            )
            
            print(f"✅ Created TemporalModel with architecture {self.architecture}")
            return model
            
        except Exception as e:
            print(f"❌ Error creating model: {e}")
            raise
    
    def _load_checkpoint(self, checkpoint_path):
        """Load pretrained weights từ checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            print(f"📦 Loading checkpoint: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_pos' in checkpoint:
                    state_dict = checkpoint['model_pos']
                    print(f"   Format: Dictionary with 'model_pos' key")
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print(f"   Format: Dictionary with 'model_state_dict' key")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print(f"   Format: Dictionary with 'state_dict' key")
                else:
                    # Assume entire dict is state dict
                    state_dict = checkpoint
                    print(f"   Format: Direct state dictionary")
            else:
                # Assume it's directly the state dict
                state_dict = checkpoint
                print(f"   Format: Direct tensor dictionary")
            
            # Load state dict
            self.model.load_state_dict(state_dict, strict=True)
            
            # Print checkpoint info
            file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
            print(f"✅ Successfully loaded checkpoint")
            print(f"   File size: {file_size:.1f} MB")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            # Print additional info if available
            if isinstance(checkpoint, dict):
                if 'epoch' in checkpoint:
                    print(f"   Epoch: {checkpoint['epoch']}")
                if 'loss' in checkpoint:
                    print(f"   Loss: {checkpoint['loss']:.4f}")
                if 'lr' in checkpoint:
                    print(f"   Learning rate: {checkpoint['lr']}")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print(f"This might be due to:")
            print(f"1. Incompatible model architecture")
            print(f"2. Corrupted checkpoint file")
            print(f"3. Mismatched PyTorch versions")
            raise
    
    def predict_3d(self, poses_2d, width, height, return_confidence=False):
        """
        Predict 3D poses từ 2D poses với FULL frames (không mất frame)
        
        Args:
            poses_2d: 2D poses array (num_frames, num_joints, 2 or 3)
            width: Image width for normalization
            height: Image height for normalization
            return_confidence: Return confidence scores if available
            
        Returns:
            3D poses array (num_frames, num_joints, 3) - SAME LENGTH as input
        """
        if len(poses_2d) == 0:
            raise ValueError("Empty poses_2d array")
        
        # Convert to numpy if needed
        if torch.is_tensor(poses_2d):
            poses_2d = poses_2d.cpu().numpy()
        poses_2d = np.array(poses_2d)
        
        # Extract x,y coordinates only
        if poses_2d.shape[-1] > 2:
            poses_2d_xy = poses_2d[..., :2]
        else:
            poses_2d_xy = poses_2d
        
        original_length = len(poses_2d_xy)  # Lưu length gốc
        
        print(f"🎯 Predicting 3D poses...")
        print(f"   Input shape: {poses_2d_xy.shape}")
        print(f"   Image size: {width}x{height}")
        
        # Normalize screen coordinates
        poses_2d_norm = normalize_screen_coordinates(poses_2d_xy, w=width, h=height)
        
        # ===== PAD INPUT TO PRESERVE ALL FRAMES =====
        pad_size = self.receptive_field // 2  # 121 frames
        
        print(f"   Padding: {pad_size} frames at start and end")
        
        # Edge padding (repeat first/last frames)
        poses_2d_padded = np.pad(
            poses_2d_norm,
            pad_width=((pad_size, pad_size), (0, 0), (0, 0)),
            mode='edge'  # Repeat edge values
        )
        
        print(f"   Padded shape: {poses_2d_padded.shape}")
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(poses_2d_padded).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output_3d = self.model(input_tensor)
        
        # Convert back to numpy
        poses_3d_padded = output_3d.squeeze(0).cpu().numpy()
        
        print(f"   Predicted shape (padded): {poses_3d_padded.shape}")
        
        # ===== EXTRACT ORIGINAL FRAMES (REMOVE PADDING) =====
        # Model output length might be: input_length - receptive_field + 1
        # We need to extract the middle section corresponding to original frames
        
        if len(poses_3d_padded) >= original_length:
            # Calculate trim indices
            # Center the extraction
            start_idx = (len(poses_3d_padded) - original_length) // 2
            end_idx = start_idx + original_length
            
            poses_3d = poses_3d_padded[start_idx:end_idx]
        else:
            # Fallback: if output is shorter (shouldn't happen with padding)
            poses_3d = poses_3d_padded
        
        print(f"   Output shape: {poses_3d.shape}")
        print(f"✅ 3D prediction completed - preserved all {original_length} frames")
        
        return poses_3d
    
    def _pad_sequence(self, poses_2d):
        """Pad sequence to match receptive field"""
        num_frames = len(poses_2d)
        
        if num_frames >= self.receptive_field:
            return poses_2d
        
        # Calculate padding
        padding_needed = self.receptive_field - num_frames
        padding_left = padding_needed // 2
        padding_right = padding_needed - padding_left
        
        # Pad with edge frames
        if num_frames == 1:
            # Special case: single frame
            poses_padded = np.repeat(poses_2d, self.receptive_field, axis=0)
        else:
            poses_padded = np.concatenate([
                np.repeat(poses_2d[:1], padding_left, axis=0),  # Repeat first frame
                poses_2d,
                np.repeat(poses_2d[-1:], padding_right, axis=0)  # Repeat last frame
            ], axis=0)
        
        return poses_padded
    
    def batch_predict_3d(self, batch_poses_2d, width, height, batch_size=8):
        """
        Predict 3D poses for a batch of sequences (for multiple people)
        
        Args:
            batch_poses_2d: List of 2D pose sequences
            width: Image width
            height: Image height
            batch_size: Processing batch size
            
        Returns:
            List of 3D pose sequences
        """
        print(f"🔄 Batch processing {len(batch_poses_2d)} sequences...")
        
        results = []
        
        for i in range(0, len(batch_poses_2d), batch_size):
            batch = batch_poses_2d[i:i+batch_size]
            batch_results = []
            
            for j, poses_2d in enumerate(batch):
                if poses_2d is not None and len(poses_2d) > 0:
                    try:
                        poses_3d = self.predict_3d(poses_2d, width, height)
                        batch_results.append(poses_3d)
                        print(f"   ✅ Sequence {i+j+1}/{len(batch_poses_2d)}: {poses_3d.shape}")
                    except Exception as e:
                        print(f"   ❌ Sequence {i+j+1}: Error - {e}")
                        batch_results.append(None)
                else:
                    print(f"   ⚠️ Sequence {i+j+1}: Empty or None")
                    batch_results.append(None)
            
            results.extend(batch_results)
        
        valid_results = sum(1 for r in results if r is not None)
        print(f"✅ Batch processing completed: {valid_results}/{len(batch_poses_2d)} successful")
        
        return results
    
    def evaluate_on_sequence(self, poses_2d, poses_3d_gt, width, height):
        """
        Evaluate model performance on a sequence với ground truth
        
        Args:
            poses_2d: 2D input poses
            poses_3d_gt: Ground truth 3D poses
            width: Image width
            height: Image height
            
        Returns:
            dict: Evaluation metrics
        """
        # Predict 3D poses
        poses_3d_pred = self.predict_3d(poses_2d, width, height)
        
        # Convert to torch tensors
        pred_tensor = torch.FloatTensor(poses_3d_pred)
        gt_tensor = torch.FloatTensor(poses_3d_gt)
        
        # Calculate metrics
        mpjpe_error = mpjpe(pred_tensor, gt_tensor).item()
        p_mpjpe_error = p_mpjpe(pred_tensor, gt_tensor).item()
        
        metrics = {
            'mpjpe': mpjpe_error,  # Mean Per Joint Position Error
            'p_mpjpe': p_mpjpe_error,  # Procrustes-aligned MPJPE
            'num_frames': len(poses_3d_pred),
            'num_joints': poses_3d_pred.shape[1]
        }
        
        print(f"📊 Evaluation metrics:")
        print(f"   MPJPE: {mpjpe_error:.2f} mm")
        print(f"   P-MPJPE: {p_mpjpe_error:.2f} mm")
        
        return metrics
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            'architecture': self.architecture,
            'receptive_field': self.receptive_field,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'input_joints': 17,
            'output_joints': 17,
            'coordinate_system': 'Human3.6M'
        }
        
        return info
    
    def print_model_summary(self):
        """Print detailed model summary"""
        info = self.get_model_info()
        
        print(f"\n📋 VideoPose3D Model Summary:")
        print(f"   Architecture: {info['architecture']}")
        print(f"   Receptive field: {info['receptive_field']} frames")
        print(f"   Input: {info['input_joints']} joints (2D)")
        print(f"   Output: {info['output_joints']} joints (3D)")
        print(f"   Parameters: {info['total_parameters']:,}")
        print(f"   Device: {info['device']}")
        print(f"   Coordinate system: {info['coordinate_system']}")

def test_videopose3d():
    """Test VideoPose3D với dummy data"""
    print("🧪 Testing VideoPose3D integration...")
    
    if not VIDEOPOSE3D_AVAILABLE:
        print("❌ VideoPose3D not available for testing")
        return False
    
    try:
        # Create predictor
        predictor = VideoPose3DPredictor()
        
        # Create dummy 2D poses
        num_frames = 100
        num_joints = 17
        width, height = 1920, 1080
        
        # Random 2D poses
        poses_2d = np.random.rand(num_frames, num_joints, 2) * np.array([width, height])
        
        print(f"🎯 Testing with dummy data:")
        print(f"   Input: {poses_2d.shape}")
        print(f"   Image size: {width}x{height}")
        
        # Predict 3D
        poses_3d = predictor.predict_3d(poses_2d, width, height)
        
        print(f"   Output: {poses_3d.shape}")
        print(f"✅ Test completed successfully!")
        
        # Print model summary
        predictor.print_model_summary()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Demo và test VideoPose3D integration"""
    print("🎬 VideoPose3D Integration Demo")
    print("=" * 50)
    
    # Check availability
    if not VIDEOPOSE3D_AVAILABLE:
        print("❌ VideoPose3D repository not found!")
        print("\n📝 Setup instructions:")
        print("1. Clone VideoPose3D repository:")
        print("   git clone https://github.com/facebookresearch/VideoPose3D.git")
        print("2. Download pretrained checkpoint:")
        print("   cd VideoPose3D/checkpoint")
        print("   wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin")
        print("3. Run this script again")
        return
    
    # Test integration
    success = test_videopose3d()
    
    if success:
        print(f"\n🎉 VideoPose3D integration working correctly!")
        print(f"📁 VideoPose3D path: {VIDEOPOSE3D_PATH}")
        print(f"🚀 Ready for 3D pose estimation!")
    else:
        print(f"\n💥 Integration test failed!")
        print(f"Please check:")
        print(f"1. VideoPose3D repository structure")
        print(f"2. Pretrained checkpoint availability")
        print(f"3. Python dependencies")

if __name__ == "__main__":
    main()