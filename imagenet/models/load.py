from .resnet import ResNet
from .resnet_ap import ResNetAP
from efficientnet_pytorch import EfficientNet


def load_model(name, depth, nclass=10):
    if name == 'resnet':
        model = ResNet(depth, num_classes=nclass, norm_type='batch')
    elif name == 'resnet_ap':
        model = ResNetAP(depth, num_classes=nclass, norm_type='instance')
    elif name == 'efficientnet':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=nclass)
    else:
        AssertionError("Check model name!")

    return model


if __name__ == "__main__":
    import torch

    name = 'efficientnet'
    depth = 10
    nclass = 10

    model = load_model(name, depth, nclass=10)
    model = model.cuda()
    print(model)

    data = torch.ones([128, 3, 256, 256]).to('cuda')
    output = model(data)
    print(output.shape)