from .segmentation_vis import SegVis


def tensorboard_factory(type_vis, conf: dict()):
    if type_vis == "segmentation":
        return SegVis(conf)
    else:
        raise Exception("Unknown vis type")
