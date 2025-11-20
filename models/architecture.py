import torch
import torch.nn as nn
from torchvision import models

class CNN_BiLSTM(nn.Module):
    def __init__(self, num_classes=7, hidden_size=128, dropout_rate=0.3):
        super(CNN_BiLSTM, self).__init__()
        
        # 1. Backbone: EfficientNet-B7
        # We load pretrained weights and remove the classifier head
        weights = models.EfficientNet_B7_Weights.DEFAULT
        self.cnn = models.efficientnet_b7(weights=weights)
        
        # Get the input dimension of the original classifier (usually 2560 for B7)
        cnn_out_features = self.cnn.classifier[1].in_features
        
        # Remove the original classification head to use it as a feature extractor
        self.cnn.classifier = nn.Identity()
        
        # 2. Temporal: Bi-Directional LSTM
        # Input size is the output of CNN (2560), Hidden size is 128 per direction
        self.lstm = nn.LSTM(
            input_size=cnn_out_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Classification Head
        # Input dim = 128 * 2 (bidirectional) = 256
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, Frames, Channels, Height, Width)
        batch_size, time_steps, C, H, W = x.size()
        
        # Flatten batch and time dimensions for CNN processing
        # cnn_input shape: (Batch * Frames, C, H, W)
        c_in = x.view(batch_size * time_steps, C, H, W)
        
        # Extract features with CNN
        # c_out shape: (Batch * Frames, 2560)
        c_out = self.cnn(c_in)
        
        # Reshape back to sequence format for LSTM
        # rnn_input shape: (Batch, Frames, 2560)
        rnn_input = c_out.view(batch_size, time_steps, -1)
        
        # LSTM Forward
        # output shape: (Batch, Frames, 2*hidden_size)
        # hn shape: (2, Batch, hidden_size) -> (num_layers * num_directions, batch, hidden_size)
        lstm_out, (hn, cn) = self.lstm(rnn_input)
        
        # Temporal Aggregation: Take the final hidden state
        # We concatenate the final forward state and the final backward state
        # hn[-2] is the last forward state, hn[-1] is the last backward state
        final_feature = torch.cat((hn[-2], hn[-1]), dim=1) # Shape: (Batch, 256)
        
        # Classification
        logits = self.head(final_feature)
        return logits
