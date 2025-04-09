import torch.nn as nn

# Vocal Separator CNN V18

class VocalSeparatorCNN(nn.Module):
    def __init__(self):
        super(VocalSeparatorCNN, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(2, 64, kernel_size=7, padding=3), nn.ReLU(), nn.Dropout(0.1))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=5, padding=2), nn.ReLU(), nn.Dropout(0.1))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.1))
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.2))

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.2))

        # Decoder
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(1024, 512, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.2))
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.1))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=5, padding=2), nn.ReLU(), nn.Dropout(0.1))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=7, padding=3), nn.ReLU())

        # Final layer
        self.final = nn.Sequential(nn.ConvTranspose2d(64, 2, kernel_size=3, padding=1), nn.Sigmoid())

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # Bottleneck
        x_bottleneck = self.bottleneck(x4)

        # Decoder with skip connections
        x = self.dec4(x_bottleneck) + x4
        x = self.dec3(x) + x3
        x = self.dec2(x) + x2
        x = self.dec1(x) + x1
        output = self.final(x)

        return output