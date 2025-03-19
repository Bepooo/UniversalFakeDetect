# from .clip_models_old import CLIPModel
from .imagenet_models import ImagenetModel
from .clip_models import CLIPModel


VALID_NAMES = [
    'Imagenet:resnet18',
    'Imagenet:resnet34',
    'Imagenet:resnet50',
    'Imagenet:resnet101',
    'Imagenet:resnet152',
    'Imagenet:vgg11',
    'Imagenet:vgg19',
    'Imagenet:swin-b',
    'Imagenet:swin-s',
    'Imagenet:swin-t',
    'Imagenet:vit_b_16',
    'Imagenet:vit_b_32',
    'Imagenet:vit_l_16',
    'Imagenet:vit_l_32',

    'CLIP:RN50', 
    'CLIP:RN101', 
    'CLIP:RN50x4', 
    'CLIP:RN50x16', 
    'CLIP:RN50x64', 
    'CLIP:ViT-B/32', 
    'CLIP:ViT-B/16', 
    'CLIP:ViT-L/14', 
    'CLIP:ViT-L/14@336px',
]





def get_model(name, top_layer='fc'):
    assert name in VALID_NAMES
    if name.startswith("Imagenet:"):
        return ImagenetModel(name[9:]) 
    elif name.startswith("CLIP:"):
        # print('getmodel函数：', top_layer)
        return CLIPModel(name[5:], top_layer=top_layer)
    else:
        assert False 
