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


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()
# net.apply(apply_dropout)


def model_train(args, net, train_loader, valid_loader):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)  # weight_decay=0.0004)
    step_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    write_log = utils.WriteLog(args.model_name)
    save_best_model = utils.SaveBestModel(args.model_name)

    criterion = lf.BCE_Dice_Loss()

    for epoch in range(1, args.epochs+1):
        print('======================== {} ========================'.format(epoch))
        for param_group in optimizer.param_groups:
            print('LR: {}'.format(param_group['lr']))

        train_loader.dataset.epoch = epoch

        ################### Train #################
        outputs_train = run_epoch(net, train_loader, optimizer, 'Training', True, criterion)
        print('\n')
        print('output train metric:')
        train_metrics = utils.compute_metrics(outputs_train['all_predictions'],
                                              outputs_train['all_targets'],
                                              outputs_train['loss_total'],
                                              )
        write_log.log_losses('train.log', epoch, outputs_train['loss_total'], train_metrics)

        ################### Valid #################
        outputs_valid = run_epoch(net, valid_loader, None, 'Validating', False, criterion)                                                                       
        print('\n')
        print('output valid metric:')
        valid_metrics = utils.compute_metrics(outputs_valid['all_predictions'],
                                              outputs_valid['all_targets'],
                                              outputs_valid['loss_total'],
                                              )
        write_log.log_losses('valid.log', epoch, outputs_valid['loss_total'], valid_metrics)

        step_scheduler.step(outputs_valid['loss_total'])

        ############## Log and Save ##############
        save_best_model.evaluate(valid_metrics, epoch, net, outputs_valid['all_predictions'], outputs_valid['all_image_ids'])


def run_epoch(net, data, optimizer, desc, train=False, criterion=None):
    if train:
        net.train()
        optimizer.zero_grad()
    else:
        net.eval()

    all_predictions = []
    all_targets = []
    all_image_ids = []
    loss_total = []

    for batch in tqdm(data, mininterval=0.5, desc=desc, leave=False, ncols=50):

        images, masks = model_input_MRI(batch)

        if train:
            pred, pred_masks = net(images.cuda(), masks.cuda())
        else:
            with torch.no_grad():
                pred, pred_masks = net(images.cuda(), masks.cuda())

        loss = criterion(pred[0], masks.cuda()) + criterion(pred[1], masks.cuda()) + criterion(pred[2], masks.cuda()) + criterion(pred[3], masks.cuda())

        loss_total.append((0.25 * loss).item())

        # print('backward...')
        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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

    net = SAMforCoarseSeg()

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
