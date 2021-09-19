import torch.nn as nn
import torchvision

class Encoder(nn.Module):
    def __init__(self, encoding_image_size = 14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoding_image_size

        resnet = torchvision.models.resnet101(pretrained=True)

        # Remove linear and pool layer (classification을 하지 않기 때문)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)