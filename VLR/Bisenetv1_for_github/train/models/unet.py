from collections import OrderedDict

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

device_num = 0

class UNet(nn.Module):

    def __init__(self, n_classes=2, mode='train',in_channels=3, init_features=32):
        super(UNet, self).__init__()
        
        # self.conf = conf
        features = init_features
        out_channels = n_classes
        self.mode = mode

        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.conv_out16 = nn.Conv2d(
            in_channels=features * 2,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )
        self.conv_out32 = nn.Conv2d(
            in_channels=features * 4,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )

    # def forward(self, x):
    def forward(self, data_dict:dict) -> dict:
        
        if self.mode == 'train':
            x = data_dict['img'].cuda(device_num)
            lb = data_dict['label'].cuda(device_num)
            lb = torch.squeeze(lb, 1)
            H, W = x.size()[2:]

            enc1 = self.encoder1(x)
            enc2 = self.encoder2(self.pool1(enc1))
            enc3 = self.encoder3(self.pool2(enc2))
            enc4 = self.encoder4(self.pool3(enc3))

            bottleneck = self.bottleneck(self.pool4(enc4))

            dec4 = self.upconv4(bottleneck)
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)
            dec1 = self.decoder1(dec1)

            feat_out = self.conv(dec1)
            feat_out_single_channel = feat_out.argmax(1)
            '''
            "feat_out.argmax(1)" is taking the biggest value in each pixel across all the different labels
            and chooses that label as the argument.
            '''
            ret_dict = {
                'out': feat_out, 
                'img': data_dict['img'], 'lb': lb,
                'out_single_channel': feat_out_single_channel,
            }
            
            # if self.training:
            # feat_out16 = feat_out
            # ret_dict['out16'] = feat_out16
            feat_out16 = self.conv_out16(dec2)
            ret_dict['out16'] = F.interpolate(
                input=feat_out16,
                size=(H, W),
                mode='bilinear',
                align_corners=True)
            
            # feat_out32 = feat_out
            # feat_out32 = self.conv_out32(dec3)
            feat_out32 = self.conv_out32(dec3)
            ret_dict['out32'] = F.interpolate(
                    input=feat_out32,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=True)

            return ret_dict
        else:
            x = data_dict['img'].cuda(device_num)
            H, W = x.size()[2:]

            enc1 = self.encoder1(x)
            enc2 = self.encoder2(self.pool1(enc1))
            enc3 = self.encoder3(self.pool2(enc2))
            enc4 = self.encoder4(self.pool3(enc3))

            bottleneck = self.bottleneck(self.pool4(enc4))

            dec4 = self.upconv4(bottleneck)
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)
            dec1 = self.decoder1(dec1)

            feat_out = self.conv(dec1)
            feat_out_single_channel = feat_out.argmax(1)
            '''
            "feat_out.argmax(1)" is taking the biggest value in each pixel across all the different labels
            and chooses that label as the argument.
            '''
            ret_dict = {
                'out': feat_out, 
                'img': data_dict['img'],
                'out_single_channel': feat_out_single_channel,
            }
            
            feat_out16 = self.conv_out16(dec2)
            ret_dict['out16'] = F.interpolate(
                input=feat_out16,
                size=(H, W),
                mode='bilinear',
                align_corners=True)
            
            feat_out32 = self.conv_out32(dec3)
            ret_dict['out32'] = F.interpolate(
                    input=feat_out32,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=True)

            return ret_dict

    def save_model(self, folder_path:str, epoch:int) -> None:

        torch.save(
        {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
        }, os.path.join(folder_path, 'unet.pth')
        )

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )