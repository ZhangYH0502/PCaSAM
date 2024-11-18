import torch
import torch.nn as nn

import argparse
import os
from tqdm import tqdm
import numpy as np

import utils1 as utils
from datamodule.MRI_CancerSeg import MRIDatamodule, model_input_MRI
from config_args import get_args
import loss_function as lf

from OurModel import *


def model_train(args, net, train_loader, valid_loader):

    criterion = lf.BCE_Dice_Loss()

    ################### Valid #################
    outputs_valid = run_epoch(net, valid_loader, 'Validating', criterion)
    print('\n')
    print('output valid metric:')
    valid_metrics = utils.compute_metrics(outputs_valid['all_predictions'],
                                          outputs_valid['all_targets'],
                                          outputs_valid['loss_total'],
                                          )

def run_epoch(net, data, desc, criterion=None):

    all_predictions = []
    all_targets = []
    all_image_ids = []
    loss_total = []

    for batch in tqdm(data, mininterval=0.5, desc=desc, leave=False, ncols=50):

        images, masks = model_input_MRI(batch)

        pred_logits, pred_masks = net(images.cuda(), masks.cuda())

        loss = criterion(pred_logits, masks.cuda())

        loss_total.append(loss.item())

        # pred = torch.sigmoid(pred)
        # pred[pred > 0.5] = 1
        # pred[pred <= 0.5] = 0

        for i in range(pred_masks.shape[0]):
            all_predictions.append(pred_masks[i, :, :].data.cpu())
            all_targets.append(masks[i, :, :].data.cpu())

        all_image_ids += batch['image_id']

    all_predictions = torch.stack(all_predictions, dim=0)
    all_targets = torch.stack(all_targets, dim=0)
    loss_total = np.mean(loss_total)

    outputs = {
        'all_predictions': all_predictions,
        'all_targets': all_targets,
        'all_image_ids': all_image_ids,
        'loss_total': loss_total,
    }

    return outputs


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    args = get_args(argparse.ArgumentParser())
    print(args.model_name)
    print('Labels: {}'.format(args.num_labels))

    train_loader = MRIDatamodule(
        data_root=args.data_root,
        image_size=(256, 256),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        mode='train',
    )
    valid_loader = MRIDatamodule(
        data_root=args.data_root,
        image_size=(256, 256),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        mode='test',
    )
    print('train_dataset len:', len(train_loader.dataset))
    print('valid_dataset len:', len(valid_loader.dataset))

    net = Customize_Model_statistic()

    model_total_params = sum(p.numel() for p in net.parameters())
    model_grad_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total params: {0}M\t Gradient Parameters: {1}M".format(model_total_params / 1e5, model_grad_params / 1e6))
    # exit(0)

    print("Using", torch.cuda.device_count(), "GPUs!")
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net = net.cuda()

    # checkpoint = torch.load(args.model_name + '/best_model.pt')
    # net.load_state_dict(checkpoint['state_dict'])
    # print('model loaded')
    # print("epoch:", checkpoint['epoch'])
    # print("valid_mDSC:", checkpoint['valid_mDSC'])

    model_train(args, net, train_loader, valid_loader)
