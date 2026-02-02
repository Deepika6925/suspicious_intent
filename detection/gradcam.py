import torch
import cv2
import numpy as np

class DummyCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, 1, 1)
        self.fc = torch.nn.Linear(16*64*64, 1)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        self.feature_map = x
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

cnn_model = DummyCNN()

def gradcam_overlay(frame, model):
    img = cv2.resize(frame, (64,64))
    t = torch.tensor(img.transpose(2,0,1)/255.0, dtype=torch.float32).unsqueeze(0)
    _ = model(t)
    fmap = model.feature_map[0].detach().mean(0).numpy()
    fmap = cv2.resize(fmap, (frame.shape[1], frame.shape[0]))
    fmap = (fmap - fmap.min()) / (fmap.max()-fmap.min()+1e-6)
    heatmap = cv2.applyColorMap(np.uint8(255*fmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    return overlay
