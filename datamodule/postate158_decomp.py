import os
import random

import cv2
import numpy as np
import nibabel as nib


def img_norm(x):
    x1 = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x1 * 255

def read_ni_to_image():
    path = 'D:\\postate\\postate158_decomp\\prostate158_train\\train'
    path_out = 'D:\\postate\\postate158_decomp\\images_total_pos\\train'

    path_adc = os.path.join(path_out, 'adc')
    if not os.path.exists(path_adc):
        os.makedirs(path_adc)
    path_dwi = os.path.join(path_out, 'dwi')
    if not os.path.exists(path_dwi):
        os.makedirs(path_dwi)
    path_t2 = os.path.join(path_out, 't2')
    if not os.path.exists(path_t2):
        os.makedirs(path_t2)
    path_label = os.path.join(path_out, 'label')
    if not os.path.exists(path_label):
        os.makedirs(path_label)
    path_label = os.path.join(path_out, 'tumor_label')
    if not os.path.exists(path_label):
        os.makedirs(path_label)

    for i in range(139):
        idx = str(i + 20)
        if len(idx) == 1:
            idx = '00' + idx
        if len(idx) == 2:
            idx = '0' + idx

        path_idx = os.path.join(path, idx)

        file_list = os.listdir(path_idx)

        for nii_name in file_list:
            if nii_name == 'adc.nii.gz':
                path_out_nii = path_adc
            elif nii_name == 'dwi.nii.gz':
                path_out_nii = path_dwi
            elif nii_name == 't2.nii.gz':
                path_out_nii = path_t2
            elif nii_name == 't2_anatomy_reader1.nii.gz':
                path_out_nii = path_label
            else:
                continue

            img = nib.load(os.path.join(path_idx, nii_name))
            img_arr = img.get_fdata()

            image_num = img_arr.shape[2]
            for j in range(image_num):
                j_idx = idx + '_' + str(j)
                if nii_name == 't2_anatomy_reader1.nii.gz':
                    output_img = img_arr[:, :, j] * 100
                else:
                    output_img = img_arr[:, :, j]
                    if np.max(output_img) == 0:
                        output_img = img_arr[:, :, j]
                    else:
                        output_img = img_norm(img_arr[:, :, j])
                cv2.imwrite(os.path.join(path_out_nii, j_idx) + '.png', output_img)

        # if 'adc_tumor_reader1.nii.gz' in file_list:
        #     img = nib.load(os.path.join(path_idx, 'adc_tumor_reader1.nii.gz'))
        # else:
        #     img = nib.load(os.path.join(path_idx, 'empty.nii.gz'))
        #
        # img_arr = img.get_fdata()
        #
        # image_num = img_arr.shape[2]
        # for j in range(image_num):
        #     j_idx = idx + '_' + str(j)
        #     output_img = img_arr[:, :, j] * 100
        #     cv2.imwrite(os.path.join(path_label, j_idx)+'.png', output_img)


def delete_negative_samples():

    path_in = '/home/amax/Public/data/Prostate-MRI-US-Biopsy/original_data'
    path_out = '/home/amax/Public/data/Prostate-MRI-US-Biopsy/images_pos'

    image_list = os.listdir(path_in)

    random.shuffle(image_list)

    for idx in range(len(image_list)):

        image_name = image_list[idx]
        try:
            subimage_list = os.listdir(os.path.join(path_in, image_name, 'mask'))
        except Exception as e:
            continue

        for subimage_name in subimage_list:

            image_adc = cv2.imread(path_in + '/' + image_name + '/' + 'adc' + '/' + subimage_name, cv2.IMREAD_GRAYSCALE)
            image_dwi = cv2.imread(path_in + '/' + image_name + '/' + 'dwi' + '/' + subimage_name, cv2.IMREAD_GRAYSCALE)
            image_t2 = cv2.imread(path_in + '/' + image_name + '/' + 't2' + '/' + subimage_name, cv2.IMREAD_GRAYSCALE)
            image_tumor = cv2.imread(path_in + '/' + image_name + '/' + 'mask' + '/' + subimage_name, cv2.IMREAD_GRAYSCALE)

            image_adc = np.array(image_adc)
            image_dwi = np.array(image_dwi)
            image_t2 = np.array(image_t2)
            image_tumor = np.array(image_tumor)

            image_tumor[image_tumor > 0] = 255

            # r = image_adc * image_dwi * image_t2 * image_tumor
            # print(r.shape)

            if np.sum(image_tumor) > 0:
                new_name = image_name + '_' + subimage_name
                if idx < 750:
                    try:
                        cv2.imwrite(path_out + '/train/' + 'adc' + '/' + new_name, image_adc)
                        cv2.imwrite(path_out + '/train/' + 'dwi' + '/' + new_name, image_dwi)
                        cv2.imwrite(path_out + '/train/' + 't2' + '/' + new_name, image_t2)
                        cv2.imwrite(path_out + '/train/' + 'label' + '/' + new_name, image_tumor)
                    except Exception as e:
                        continue
                else:
                    try:
                        cv2.imwrite(path_out + '/test/' + 'adc' + '/' + new_name, image_adc)
                        cv2.imwrite(path_out + '/test/' + 'dwi' + '/' + new_name, image_dwi)
                        cv2.imwrite(path_out + '/test/' + 't2' + '/' + new_name, image_t2)
                        cv2.imwrite(path_out + '/test/' + 'label' + '/' + new_name, image_tumor)
                    except Exception as e:
                        continue


if __name__ == '__main__':
    # read_ni_to_image()
    delete_negative_samples()

