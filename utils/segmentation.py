import torch
from torchvision import models, transforms
import cv2
import numpy as np

class Segmenter:
    def __init__(self, device='cuda'):
        self.device = device
        # Using DeepLabV3 with ResNet101 backbone
        self.model = models.segmentation.deeplabv3_resnet101(weights='DEFAULT').to(device)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_foreground(self, frame):
        # Preprocess input
        input_tensor = self.preprocess(frame).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        # Class 15 is usually 'person' in COCO/Pascal VOC
        # Assuming the pretrained model detects person at index 15
        output_predictions = output.argmax(0)
        
        # Create binary mask for 'person' class (Adjust index based on specific pretrained model)
        # Here we treat predicted class 15 as person. 
        mask = (output_predictions == 15).byte().cpu().numpy()
        
        # Resize mask to original frame size if needed, or resize frame to mask
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Apply mask
        foreground = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Set background to black or gray if needed
        return foreground
