import torch
import torch.nn as nn

from utils.prep_utils import MODEL_NEURONS


class ConvBlock(nn.Module):
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_depth),
            nn.Conv2d(in_depth, out_depth, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_depth),
            nn.Conv2d(out_depth, out_depth, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ShallowUNet(nn.Module):
    """
    Implementation of UNet, slightly modified:
    - less downsampling blocks
    - less neurons in the layers
    - Batch Normalization added
    
    Link to paper on original UNet:
    https://arxiv.org/abs/1505.04597
    """
    
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv_down1 = ConvBlock(in_channel, MODEL_NEURONS)
        self.conv_down2 = ConvBlock(MODEL_NEURONS, MODEL_NEURONS * 2)
        self.conv_down3 = ConvBlock(MODEL_NEURONS * 2, MODEL_NEURONS * 4)
        self.conv_bottleneck = ConvBlock(MODEL_NEURONS * 4, MODEL_NEURONS * 8)

        self.maxpool = nn.MaxPool2d(2)
        self.upsamle = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv_up1 = ConvBlock(
            MODEL_NEURONS * 8 + MODEL_NEURONS * 4, MODEL_NEURONS * 4
        )
        self.conv_up2 = ConvBlock(
            MODEL_NEURONS * 4 + MODEL_NEURONS * 2, MODEL_NEURONS * 2
        )
        self.conv_up3 = ConvBlock(MODEL_NEURONS * 2 + MODEL_NEURONS, MODEL_NEURONS)

        self.conv_out = nn.Sequential(
            nn.Conv2d(MODEL_NEURONS, out_channel, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        conv_d1 = self.conv_down1(x)
       # print("lvl 0 down: " , conv_d1.shape)

        conv_d2 = self.conv_down2(self.maxpool(conv_d1))
      #  print("lvl 1 down: " , conv_d2.shape)
#
        conv_d3 = self.conv_down3(self.maxpool(conv_d2))
       # print("lvl 2 down: " , conv_d3.shape)
        conv_b = self.conv_bottleneck(self.maxpool(conv_d3))

       # print("lvl 2 up: " , conv_b.shape)
  
        conv_u1 = self.conv_up1(torch.cat([self.upsamle(conv_b), conv_d3], dim=1))
       # print("lvl 1 up: " , conv_u1.shape)

        conv_u2 = self.conv_up2(torch.cat([self.upsamle(conv_u1), conv_d2], dim=1))
        #print("lvl 0 up: " , conv_u2.shape)

        conv_u3 = self.conv_up3(torch.cat([self.upsamle(conv_u2), conv_d1], dim=1))

        out = self.conv_out(conv_u3)
        return out
    
    
class LeveledUNet(nn.Module):
    def __init__(self, in_channels, out_channels, levels):
        super().__init__()

        self.levels = levels
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.maxpool = nn.MaxPool2d(2)

        # Create encoding blocks
        self.conv_down = nn.ModuleList()
        #down_neurons = []
        for level in range(levels):
            input_channels = in_channels if level == 0 else MODEL_NEURONS * (2 ** (level - 1))
            output_channels = MODEL_NEURONS * (2 ** level)
            cb = ConvBlock(input_channels, output_channels)
            #down_neurons.append(output_channels)
            self.conv_down.append(cb)
          #  print(f"Level down {level}", cb)

        # Create bottleneck block
        self.conv_bottleneck = ConvBlock(MODEL_NEURONS * (2 ** (levels - 1)), MODEL_NEURONS * (2 ** levels))

        # Create decoding blocks with skip connections
        self.conv_up = []
        for level in reversed(range(levels)):
            input_channels = MODEL_NEURONS * (2 ** level)+MODEL_NEURONS * (2 ** (level + 1))
            output_channels = MODEL_NEURONS * (2 ** level)
            cb = ConvBlock(input_channels, output_channels)
            self.conv_up.append(cb)
           # print(f"Level up {level}", cb, input_channels, output_channels)
    
        self.conv_up = nn.ModuleList(reversed(self.conv_up)) # FIFO
        # Output convolution
        self.conv_out = nn.Sequential(
            nn.Conv2d(MODEL_NEURONS, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoding_outputs = []

        # Encoding
        for level in range(self.levels):
           # print(level, "down",  x.shape)

            x = self.conv_down[level](x)
            encoding_outputs.append(x)

            x = self.maxpool(x)

        # Bottleneck
        x = self.conv_bottleneck(x)

        # Decoding with skip connections
        for level in reversed(range(self.levels)):

            x = self.upsample(x)#nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = torch.cat([x, encoding_outputs[level]], dim=1)
           # print(level,"up " ,  x.shape)

            x = self.conv_up[level](x)

        # Output
        out = self.conv_out(x)
        return out
    
    
class SimpleCNN(nn.Module):
    
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        
        self.conv1 = ConvBlock(in_channel, out_channel)
        self.conv2 = ConvBlock(out_channel, out_channel)
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        
        self.model = torch.nn.Sequential(self.conv1,self.conv2, self.conv_out)
        
    def forward(self, x):
        return self.model.forward(x)