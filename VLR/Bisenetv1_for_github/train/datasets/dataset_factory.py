import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from datasets.celebamask_hq_dataset import CelebAMask_HQ
from datasets.facesynthetic_dataset import FaceSynthetic
from datasets.P3M_dataset import P3M


def dataset_factory(type_dataset: str(), conf: dict()):
    if type_dataset == "CelebAMask_HQ":
        return (CelebAMask_HQ(conf['cfg']), None)
    elif type_dataset == "FaceSynthetic":
        return (FaceSynthetic(conf['cfg']), None)
    elif type_dataset == "P3M":
        return (P3M(conf['cfg']), None)
    else:
        raise Exception("Unknown dataset type")
