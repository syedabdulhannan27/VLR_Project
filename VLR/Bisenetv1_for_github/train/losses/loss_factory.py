from .segmentation_loss import SegmentationLoss_nn


def loss_factory(type_loss: str(), conf: dict()):
    if type_loss == "OhemCELoss":
        return SegmentationLoss_nn(conf)
    else:
        raise Exception("Unknown loss type")
