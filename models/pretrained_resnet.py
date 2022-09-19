from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torch import nn
from base_model import ModelConstructor


def get_IN_resnet(params, depth, pretrained):

    if depth == "18":
        model = resnet18(pretrained=pretrained)
    elif depth == "34":
        model = resnet34(pretrained=pretrained)
    elif depth == "50":
        model = resnet50(pretrained=pretrained)
    elif depth == "101":
        model = resnet101(pretrained=pretrained)
    elif depth == "152":
        model = resnet152(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, params["num_classes"])

    model = ModelConstructor(model, params)

    return model
