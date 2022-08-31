import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CAM_backprop(nn.Module):
    def __init__(self, weight_param, class_idx):
        super(CAM_backprop, self).__init__()
        self.weight_param = nn.Parameter(weight_param)
        self.class_idx = class_idx

    def forward(self, feature_map):

        n, c, h, w = feature_map.shape

        cam = self.weight_param[self.class_idx].view(512, 1) * feature_map.reshape((c, h*w)).to(device)
        cam = torch.sum(cam, 0)
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        return cam.view(7, 7)