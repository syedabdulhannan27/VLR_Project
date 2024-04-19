import torch
import train.confs.config_factory as c_f
from train_terminal import train_run

project_root = '/home/ishan/projects/patch_seg/VLR_Project/VLR/Bisenetv1_for_github'
type_dataset = 'P3M'    # CelebAMask_HQ | FaceSynthetic
type_model = 'bisenetv1'  # bisenetv1 unet
type_task = 'segmentation'
type_loss = 'OhemCELoss'
type_scheduler = 'ExponentialLR'
type_saver = 'standard_seg'
load_from_snapshot = False
resume_path = '/home/ishan/Bisenetv1_for_github/res/terminal_test/Unet_1.pth'
tb_name = 'bisenet_celeb_a_2' # Unet_2

device_num = 0
cfg_dict = dict(
    type_model=type_model,
    type_dataset=type_dataset,
    )
cfg = c_f.config_factory(cfg_dict)

dataset_dict = dict(
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

general_dict = dict(
    type_dataset=type_dataset,
    type_model=type_model,
    type_task=type_task,
    type_loss=type_loss,
    type_scheduler=type_scheduler,
    type_vis=type_task,
    type_saver=type_saver,
    cfg=cfg,
    training_dict=cfg['training_dict'],
    dataset_dict=dataset_dict,
    device=torch.device(f"cuda:{device_num}"),
    project_root=project_root,
    snapshot_path=f'{project_root}/res/snapshots/{tb_name}/snapshot.pt',
    model_path=f'{project_root}/res/terminal_test',
    tb_path=f'{project_root}/runs',
    tb_name=tb_name,
    load_from_snapshot=load_from_snapshot,
    resume_path=resume_path
)

# Run the training loop
train_run(general_dict)