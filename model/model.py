class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                )
        return block
    
    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3, padding = 1, output_padding = 1):
            block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=padding),
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=padding, output_padding=output_padding)
                    )
            return  block
    
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3, padding = 1):
            block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=padding),
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=padding),
                    torch.nn.Sigmoid(),
                    )
            return  block

    def Bottle_neck(self, layer_in, layer_mid, layer_out, kernel_size=3,padding = 1, output_padding = ):
            block = torch.nn.Sequential(
                            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=layer_in[0], out_channels=layer_in[1], padding=padding),
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm2d(layer_in[1]),
                            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=layer_mid[0], out_channels=layer_mid[1], padding=padding),
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm2d(layer_mid[1]),
                            torch.nn.ConvTranspose2d(in_channels=layer_out[0], out_channels=layer_out[1], kernel_size=kernel_size, stride=2, padding=padding, output_padding=output_padding)
                            )
            return  block
    
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        #Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(64, 128)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(128, 256)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode4 = self.contracting_block(256, 512)
        self.conv_maxpool4 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        self.bottleneck = self.Bottle_neck((512, 1024), (1024, 1024), (1024, 512), kernel_size = 3)
        # Decode
        self.conv_decode4 = self.expansive_block(1024, 512, 256)
        self.conv_decode3 = self.expansive_block(512, 256, 128)
        self.conv_decode2 = self.expansive_block(256, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)
        
    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)
    
    def forward(self, x):
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4  = self.conv_maxpool4(encode_block4)
        bottleneck1 = self.bottleneck(encode_pool4)
        decode_block4 = self.crop_and_concat(bottleneck1, encode_block4, crop=True)
        cat_layer3 = self.conv_decode4(decode_block4)
        decode_block3 = self.crop_and_concat(cat_layer3, encode_block3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        final_layer = self.final_layer(decode_block1)
        return  final_layer