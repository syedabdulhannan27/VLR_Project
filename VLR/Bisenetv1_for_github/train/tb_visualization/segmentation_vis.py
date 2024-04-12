import os
from torch.utils.tensorboard import SummaryWriter


class SegVis():
    def __init__(self, conf: dict()):
        self.writer_path = os.path.join(
            conf['tb_path'],
            conf['type_model'],
            conf['tb_name']
            )
        self.writer = SummaryWriter(self.writer_path)

    def update(self, info_dict: dict()):
        # info_dict = {
    #                 'im': im,
    #                 'lb': lb,
    #                 'pred': pred_dict,
    #                 'loss': loss,
    #                 'data_dict': data_dict,
    #                 'lr': scheduler.get_last_lr()[0],
    #                 'iteration': iteration
    #                 }

        global_step = info_dict['iteration']
        loss = info_dict['loss']
        lr = info_dict['lr']

        self.writer.add_scalar('Training Loss', loss, global_step=global_step)
        self.writer.add_scalar('Current LR', lr, global_step=global_step)
