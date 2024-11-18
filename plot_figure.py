import os
import numpy as np
import cv2


def show_mask(img, mask, color, alpha=0.6):
    mask = np.stack([mask, mask, mask], -1)
    img = np.where(mask, img * (1-alpha) + color * alpha, img)
    return img.astype(np.uint8)


if __name__ == '__main__':
    image_adc = '/home/amax/Public/data/postate/postate158_decomp/images_total_pos/all/adc'
    image_dwi = '/home/amax/Public/data/postate/postate158_decomp/images_total_pos/all/dwi'
    image_t2 = '/home/amax/Public/data/postate/postate158_decomp/images_total_pos/all/t2'

    path_out = '/home/amax/Public/MedSAM-main/results/postate158.OurModel.CrossFuseWithoutBox'

    image_gt = path_out + '/' + 'result_masks'
    # image_pred = '/home/amax/Public/MedSAM-main/results/MRI.OurModel.c3.1024.FeatureFuseModuleTTA'

    path_out_adc = path_out + '/' + 'gt_adc'
    if not os.path.exists(path_out_adc):
        os.makedirs(path_out_adc)
    path_out_dwi = path_out + '/' + 'gt_dwi'
    if not os.path.exists(path_out_dwi):
        os.makedirs(path_out_dwi)
    path_out_t2 = path_out + '/' + 'gt_t2'
    if not os.path.exists(path_out_t2):
        os.makedirs(path_out_t2)

    image_list = os.listdir(image_gt)

    color = np.array([30, 252, 251])

    for image_name in image_list:
        if not os.path.exists(image_adc + '/' + image_name):
            continue

        mask = cv2.imread(image_gt + '/' + image_name, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = np.array(mask)
        mask[mask > 0] = 1

        image = cv2.imread(image_adc + '/' + image_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        image = show_mask(image, mask, color)
        cv2.imwrite(path_out_adc + '/' + image_name, image)

        image = cv2.imread(image_dwi + '/' + image_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        image = show_mask(image, mask, color)
        cv2.imwrite(path_out_dwi + '/' + image_name, image)

        image = cv2.imread(image_t2 + '/' + image_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        image = show_mask(image, mask, color)
        cv2.imwrite(path_out_t2 + '/' + image_name, image)
