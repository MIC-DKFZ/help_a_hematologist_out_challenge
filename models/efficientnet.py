from efficientnet_pytorch import EfficientNet
from base_model import BaseModel
from torch import nn
from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l


class BaseEfficientNet(BaseModel):
    def __init__(self, num_classes=10, hypparams={}, net_type="l2"):
        super(BaseEfficientNet, self).__init__(hypparams)

        # EfficientNet
        self.network = EfficientNet.from_name("efficientnet-" + net_type, image_size=32, num_classes=num_classes)

    def forward(self, x):
        out = self.network(x)
        return out


def get_EfficientNetv2s(num_classes=10,pretrained=True, hypparams={}):
    if pretrained:
        model = efficientnet_v2_s(weights="EfficientNet_V2_S_Weights.DEFAULT")
    else:
        model = efficientnet_v2_s()
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

    return model

def get_EfficientNetv2m(num_classes=10,pretrained=True, hypparams={}):
    if pretrained:
        model = efficientnet_v2_m(weights="EfficientNet_V2_M_Weights.DEFAULT")
    else:
        model = efficientnet_v2_m()
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

    return model

def get_EfficientNetv2l(num_classes=10,pretrained=True, hypparams={}):
    if pretrained:
        model = efficientnet_v2_l(weights="EfficientNet_V2_L_Weights.DEFAULT")
    else:
        model = efficientnet_v2_l()
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

    return model

def EfficientNetL2(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, "l2")


def EfficientNetB8(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, "b8")


def EfficientNetB7(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, "b7")


def EfficientNetB6(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, "b6")


def EfficientNetB5(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, "b5")


def EfficientNetB4(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, "b4")


def EfficientNetB3(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, "b3")


def EfficientNetB2(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, "b2")


def EfficientNetB1(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, "b1")


def EfficientNetB0(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, "b0")
