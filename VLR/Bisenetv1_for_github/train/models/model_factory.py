from .bisenetv2 import BiSeNetV2
from .bisenetv1 import BiSeNet
from .bisenetv1_og import BiSeNet_og
from train.models.unet import UNet
from train.models.unet_v2 import UNet_v2


def model_factory(type_model: str(), conf: dict()):
    if type_model == "bisenetv1":
        return BiSeNet(conf['cfg']['n_cats'])
    elif type_model == "bisenetv1_og":
        return BiSeNet_og(conf['cfg']['n_cats'])
    elif type_model == "bisenetv2":
        return BiSeNetV2(conf['cfg']['n_cats'])
    if type_model == "unet":
        return UNet(conf['cfg']['n_cats'])
    if type_model == "unet_v2":
        return UNet_v2(conf['cfg']['n_cats'])
    else:
        raise Exception("Unknown dataset type")
