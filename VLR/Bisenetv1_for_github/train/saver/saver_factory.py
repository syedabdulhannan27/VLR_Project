from .standard_saver import SaverObject


def save_factory(type_saver: str(), conf: list()):
    general_dict = conf[0]
    # info_dataset_dict = conf[1]
    if type_saver == "standard_seg":
        return SaverObject(general_dict)
    else:
        raise Exception("Unknown saver type")
