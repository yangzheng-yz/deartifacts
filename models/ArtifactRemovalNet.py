import torch
import torch.nn as nn
# 定义主网络结构
class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        return x

# 定义用于学习加权系数的网络
class CoeffNet(nn.Module):
    def __init__(self):
        super(CoeffNet, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 128)

    def forward(self, x):
        x = x.view(x.size(0), 128, -1)  # Flatten the spatial dimensions
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.Softmax(dim=1)(x)  # Softmax to get the coefficients
        return x.view(x.size(0), 128, 1, 1)  # Reshape back

# 定义整个模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.main_net = MainNet()
        self.coeff_net = CoeffNet()

    def forward(self, x):
        features = self.main_net(x)
        coeffs = self.coeff_net(features)
        out = features * coeffs  # Element-wise multiplication
        out = out.sum(dim=1, keepdim=True)  # Sum along the channel dimension
        return out
    
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Decoder
        self.upconv3 = self.upconv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        
        self.dec3 = self.conv_block(512, 256)  # Note the change in channel size
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.maxpool = nn.MaxPool2d(2)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        # print("enc1: ", enc1.size())
        enc2 = self.enc2(self.maxpool(enc1))
        # print("enc2: ", enc2.size())
        enc3 = self.enc3(self.maxpool(enc2))
        # print("enc3: ", enc3.size())
        enc4 = self.enc4(self.maxpool(enc3))
        # print("enc4: ", enc4.size())
        
        # Decoder with skip connections
        dec3 = self.dec3(torch.cat([self.upconv3(enc4), enc3], dim=1))
        # print("dec3: ", dec3.size())
        dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc2], dim=1))
        # print("dec2: ", dec2.size())
        dec1 = self.dec1(torch.cat([self.upconv1(dec2), enc1], dim=1))
        # print("dec1: ", dec1.size())
        
        out = self.final_conv(dec1)
        # print("out: ", out.size())
        
        return out

class SimpleNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SimpleNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        
        out = self.final_conv(enc1)
        # print("out: ", out.size())
        
        return out


class RemovalNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(RemovalNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Decoder
        self.upconv3 = self.upconv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        
        self.dec3 = self.conv_block(512, 256)  # Note the change in channel size
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)
        
        self.art_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.final_conv = nn.Conv2d(2, 1, kernel_size=1)

        self.maxpool = nn.MaxPool2d(2)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.maxpool(enc1))
        enc3 = self.enc3(self.maxpool(enc2))
        enc4 = self.enc4(self.maxpool(enc3))
        
        # Decoder with skip connections
        dec3 = self.dec3(torch.cat([self.upconv3(enc4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upconv1(dec2), enc1], dim=1))
                
        artifacts = self.art_conv(dec1)
        
        out = self.final_conv(torch.cat([artifacts, x], dim=1))
        
        return out
if __name__ == "__main__":
    x = torch.randn([2,1,512,640])
    model = UNet()
    out = model(x)