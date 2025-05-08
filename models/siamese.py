import torch.nn as nn

class SiameseNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x1, x2):
        out1 = self.encoder(x1)
        out2 = self.encoder(x2)
        return out1, out2