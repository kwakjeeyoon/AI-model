import torch.nn as nn
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3),
        nn.ReLU(inplace=True)
    )
# downsampling
self.dconv_down1 = double_conv(3,64)
self.maxpool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
self.dconv_down2 = double_conv(64,128)
self.maxpool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
self. dconv_down3 = double_conv(128,256)
self.maxpool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
self.dconv_down4 = double_conv(256, 512)
self.maxpool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
self.dconv_down5 = double_conv(512, 1024)

# upsampling - 위와 정확히 대응되는 해상도 + channel
self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024,out_channels=512, kernel_size=2,stride=2)
self.up_conv_1 = double_conv(1024,512)
self.up_trans_2 = nn.ConvTranspose2d(in_channels=512,out_channels=256, kernel_size=2,stride=2)
self.up_conv_2 = double_conv(512,256)
self.up_trans_3 = nn.ConvTranspose2d(in_channels=256,out_channels=128, kernel_size=2,stride=2)
self.up_conv_3 = double_conv(256,128)
self.up_trans_4 = nn.ConvTranspose2d(in_channels=128,out_channels=64, kernel_size=2,stride=2)
self.up_conv_4 = double_conv(128,64)
self.out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

# 질문 : 위에서 나오는 conv 부분이 upsample + conv에서 conv 부분인건가?
# 만약 맞다면 NN, bilinear interpolation은 어디서 사용하는 것인가? (CV 4강 참조)