import os
import torch


class SaverObject():
    def __init__(self, general_dict: dict()) -> None:
        self.general_dict = general_dict
        self.respth = general_dict['model_path']

    def save(self, model, optimizer, crrnt_loss, epoch):
        if epoch % self.general_dict['training_dict']['save_every'] == 0:
            save_pth_basename = os.path.join(
                self.respth,
                f'{self.general_dict["tb_name"]}'
                )

            if not os.path.exists(save_pth_basename):
                os.makedirs(save_pth_basename)

            save_pth = os.path.join(
                save_pth_basename,
                f'{epoch}.pth'
                )

            state = model.state_dict()
            torch.save(state, save_pth)
            print(f'Model checkpoint saved at epoch: {epoch}')
        else:
            pass
