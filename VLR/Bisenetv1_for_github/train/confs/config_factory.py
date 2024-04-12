import json
import os


def import_json(cur_dir, type_object, object_folder):
    try:
        with open(f'{cur_dir}/{object_folder}/{type_object}_config.json') as file:
            file_dict = json.load(file)
            return file_dict
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing config file for {type_object}") from e
    except Exception as e:
        raise Exception(f"Failed to load config for {type_object} : {e}") from e


def config_factory(config_dict) -> dict:
    type_model = config_dict['type_model']
    type_dataset = config_dict['type_dataset']
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    model_cfg_dict = import_json(
        cur_dir,
        type_object=type_model,
        object_folder='models'
        )
    dataset_cfg_dict = import_json(
        cur_dir,
        type_object=type_dataset,
        object_folder='datasets'
        )
    training_cfg_dict = import_json(
        cur_dir,
        type_object='training',
        object_folder='training'
        )

    cfg_dict = dict(
        model_cfg_dict,
        **dataset_cfg_dict,
        training_dict=training_cfg_dict
        )
    return cfg_dict
