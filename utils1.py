import torch
import numpy as np
import os
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


def dice_coeff(all_predictions, all_targets, smooth=1e-5):

    pred = all_predictions.clone() == 1
    targ = all_targets.clone() == 1

    pred = pred.contiguous().view(-1).numpy()
    targ = targ.contiguous().view(-1).numpy()

    # pred = pred.contiguous().view(batch_num, -1).numpy()
    # targ = targ.contiguous().view(batch_num, -1).numpy()
    #
    # false_idx = (np.sum(targ, axis=-1) > 0)
    #
    # pred = pred[false_idx, :]
    # targ = targ[false_idx, :]

    intersection = (pred * targ).sum()

    union = pred.sum() + targ.sum()

    dsc = (2. * intersection + smooth) / (union + smooth)

    return dsc


def compute_metrics(all_predictions, all_targets, loss=None):

    total_mean = dice_coeff(all_predictions, all_targets)

    if loss is not None:
        print('loss: {:0.3f}'.format(loss))
    print('total_mDSC: {:0.1f}'.format(total_mean * 100))

    metrics_dict = {}
    metrics_dict['loss'] = loss
    metrics_dict['total_mDSC'] = total_mean

    return metrics_dict


class WriteLog:
    def __init__(self, model_name):
        self.model_name = model_name
        open(model_name+'/train.log', "w").close()
        open(model_name+'/valid.log', "w").close()

    def log_losses(self, file_name, epoch, loss, metrics):
        log_file = open(self.model_name+'/'+file_name, "a")
        log_file.write(str(epoch) + ',  ' + str(round(loss, 4)) + ',  '
                       + str(round(metrics['total_mDSC'], 4)) + '\n'
                       )
        log_file.close()


class SaveBestModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.best_mDSC = 0

    def evaluate(self, valid_metrics, epoch, model, all_preds, all_ids):

        if valid_metrics['total_mDSC'] > self.best_mDSC:
            self.best_mDSC = valid_metrics['total_mDSC']

            print('> Saving Model\n')
            save_dict = {
                'epoch': epoch,
                'state_dict': model.box_generater.state_dict(),
                # 'state_dict_f2': model.feature_fusion_for_prompt_seg.state_dict(),
                # 'state_dict_mlp': model.box_generater.state_dict(),
                'valid_mDSC': valid_metrics['total_mDSC'],
            }
            torch.save(save_dict, self.model_name + '/best_model.pt')

            print('\n')
            print('best total_mDSC:  {:0.1f}'.format(valid_metrics['total_mDSC'] * 100))

            mask_save_path = self.model_name + "/" + "result_masks"
            if not os.path.exists(mask_save_path):
                os.makedirs(mask_save_path)

            all_preds = all_preds.numpy()

            for i in range(all_preds.shape[0]):
                img = all_preds[i, :, :] * 200

                img_id = all_ids[i]

                im = Image.fromarray(np.uint8(img))
                im.save(mask_save_path + "/" + img_id)


if __name__ == "__main__":

    x = np.random.rand(3, 5)
    pred = (x > 0.6).astype(int)
    targ = (x > 0.3).astype(int)

    print(pred)
    print(targ)

    intersection = (pred * targ).sum(-1)
    print(intersection)

    union = pred.sum(-1) + targ.sum(-1)
    print(union)

    dsc = (2. * intersection + 1e-5) / (union + 1e-5)
    print(dsc)




