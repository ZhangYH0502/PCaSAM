import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
from tqdm import tqdm
import numpy as np
from PIL import Image

import utils1 as utils
from datamodule.MRI_CancerSeg import MRIDatamodule, model_input_MRI
from config_args import get_args

from OurModel import SAM_For_Test


def write_images(mask_save_path, all_preds, all_ids):
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path)

    for i in range(all_preds.shape[0]):
        img = all_preds[i, :, :].numpy() * 200
        img_id = all_ids[i]
        im = Image.fromarray(np.uint8(img))
        im.save(mask_save_path + "/" + img_id)


def model_test(args, net, train_loader, valid_loader):

    # outputs_train = run_epoch(net, train_loader, optimizer, 'Training', True, criterion)
    # print('\n')
    # print('output train metric:')
    # train_metrics = utils.compute_metrics(outputs_train['all_predictions'],
    #                                       outputs_train['all_targets'],
    #                                       outputs_train['loss_total'],
    #                                       )

    ################### Valid #################
    outputs_valid = run_epoch(net, valid_loader, 'Validating')
    print('\n')
    print('output valid metric:')
    valid_metrics = utils.compute_metrics(
        outputs_valid['all_predictions'],
        outputs_valid['all_targets'],
    )

    # valid_metrics_unet = utils.compute_metrics(
    #     outputs_valid['all_predictions_unet'],
    #     outputs_valid['all_targets'],
    # )

    ############## Log and Save ##############
    # write_images("results/test/test_adc/box/preds", outputs_valid['all_predictions'], outputs_valid['all_image_ids'])
    # write_images("results/test/test_adc/box/preds_unet", outputs_valid['all_predictions_unet'],
    #              outputs_valid['all_image_ids'])
    # write_images("results/test/test_adc/box/targs", outputs_valid['all_targets'], outputs_valid['all_image_ids'])


def run_epoch(net, data, desc):
    net.eval()

    all_predictions = []
    all_predictions_unet = []
    all_targets = []
    all_image_ids = []

    n = 1000000

    for batch in tqdm(data, mininterval=0.5, desc=desc, leave=False, ncols=50):

        images, masks = model_input_MRI(batch)
        if torch.sum(masks) == 0:
            continue

        with torch.no_grad():
            pred = net(images.cuda(), masks.cuda())

        all_predictions.append(pred[0, 0, :, :].data.cpu())
        all_targets.append(masks[0, :, :].data.cpu())

        all_image_ids += batch['image_id']

    all_predictions = torch.stack(all_predictions, dim=0)
    # all_predictions_unet = torch.stack(all_predictions_unet, dim=0)
    all_targets = torch.stack(all_targets, dim=0)

    outputs = {
        'all_predictions': all_predictions,
        'all_predictions_unet': all_predictions_unet,
        'all_targets': all_targets,
        'all_image_ids': all_image_ids,
    }

    return outputs


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    args = get_args(argparse.ArgumentParser())
    print(args.model_name)
    print('Labels: {}'.format(args.num_labels))

    train_loader = MRIDatamodule(
        data_root=args.data_root,
        image_size=(256, 256),
        batch_size=1,
        shuffle=False,
        drop_last=False,
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

    net = SAM_For_Test()

    # print("Using", torch.cuda.device_count(), "GPUs!")
    # if torch.cuda.device_count() > 1:
    #     net = nn.DataParallel(net)
    # net = net.cuda()

    model_test(args, net, train_loader, valid_loader)
