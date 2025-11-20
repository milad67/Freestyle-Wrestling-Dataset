import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from utils.segmentation import Segmenter

class FSWDataset(Dataset):
    def __init__(self, root_dir, num_frames=15, transform=None, use_segmentation=True):
        """
        Args:
            root_dir (string): Directory with all the video classes folders.
            num_frames (int): Number of frames to sample (T=15).
            use_segmentation (bool): Whether to apply DeepLabV3+ foreground extraction.
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.use_segmentation = use_segmentation
        self.clips = []
        
        if self.use_segmentation:
            print("Initializing Segmenter...")
            self.segmenter = Segmenter(device='cuda' if torch.cuda.is_available() else 'cpu')

        # Load file paths
        for cls_idx, cls_name in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls_name)
            for vid_name in os.listdir(cls_path):
                self.clips.append((os.path.join(cls_path, vid_name), cls_idx))

    def __len__(self):
        return len(self.clips)

    def sample_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        frames = []
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Apply Segmentation if enabled
                if self.use_segmentation:
                    frame = self.segmenter.get_foreground(frame)
                
                # Resize to 224x224 for EfficientNet
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
        
        cap.release()
        
        # Padding if video is too short
        while len(frames) < self.num_frames:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
        return np.array(frames)

    def __getitem__(self, idx):
        video_path, label = self.clips[idx]
        frames = self.sample_frames(video_path) # Shape: (T, H, W, C)
        
        # Convert to Torch Tensor: (T, C, H, W)
        frames = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0
        
        # Normalize (ImageNet stats)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        frames = torch.stack([normalize(f) for f in frames])
        
        return frames, label
