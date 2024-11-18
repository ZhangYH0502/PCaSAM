import os
import cv2
import numpy as np
import nibabel as nib
import SimpleITK as sitk


def img_norm(x):
    x1 = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x1 * 255

def read_ni_to_image():
    path = '/home/amax/Public/data/postate/picai_decomp'
    path_out = '/home/amax/Public/data/postate/picai_decomp_images_all/train'

    path_adc = os.path.join(path_out, 'adc')
    if not os.path.exists(path_adc):
        os.makedirs(path_adc)
    path_dwi = os.path.join(path_out, 'dwi')
    if not os.path.exists(path_dwi):
        os.makedirs(path_dwi)
    path_t2 = os.path.join(path_out, 't2')
    if not os.path.exists(path_t2):
        os.makedirs(path_t2)
    path_label = os.path.join(path_out, 'tumor_label')
    if not os.path.exists(path_label):
        os.makedirs(path_label)

    patients_list = os.listdir(path)

    for tt in range(180):
        tt1 = tt + 0
        idx = patients_list[tt1]

        path_idx = os.path.join(path, idx)

        img_adc_name = idx + '_adc_new.mha'
        img_adc = sitk.ReadImage(os.path.join(path_idx, img_adc_name))
        img_adc = np.array(sitk.GetArrayFromImage(img_adc))

        img_dwi_name = idx + '_hbv_new.mha'
        img_dwi = sitk.ReadImage(os.path.join(path_idx, img_dwi_name))
        img_dwi = np.array(sitk.GetArrayFromImage(img_dwi))

        img_t2_name = idx + '_t2w.mha'
        img_t2 = sitk.ReadImage(os.path.join(path_idx, img_t2_name))
        img_t2 = np.array(sitk.GetArrayFromImage(img_t2))

        label_name_1 = idx + '_1.nii.gz'
        label_1 = nib.load(os.path.join(path_idx, label_name_1))
        label_1 = label_1.get_fdata()

        label_name_2 = idx + '_2.nii.gz'
        if os.path.exists(os.path.join(path_idx, label_name_2)):
            label_2 = nib.load(os.path.join(path_idx, label_name_2))
            label_2 = label_2.get_fdata()
            label_1 = label_1 + label_2

        label_name_3 = idx + '_3.nii.gz'
        if os.path.exists(os.path.join(path_idx, label_name_3)):
            label_3 = nib.load(os.path.join(path_idx, label_name_3))
            label_3 = label_3.get_fdata()
            label_1 = label_1 + label_3

        label_1[label_1 > 0] = 1
        label_1 = label_1 * 200

        for num in range(label_1.shape[2]):
            slice_adc = img_adc[num, :, :]
            if np.max(slice_adc) > 0:
                slice_adc = img_norm(slice_adc)
            else:
                continue

            slice_dwi = img_dwi[num, :, :]
            if np.max(slice_dwi) > 0:
                slice_dwi = img_norm(slice_dwi)
            else:
                continue

            slice_t2 = img_t2[num, :, :]
            if np.max(slice_t2) > 0:
                slice_t2 = img_norm(slice_t2)
            else:
                continue

            labels = label_1[:, :, num]

            idx_num = idx + '_' + str(num)

            # if np.max(labels) > 0:
            cv2.imwrite(os.path.join(path_adc, idx_num) + '.png', slice_adc)
            cv2.imwrite(os.path.join(path_dwi, idx_num) + '.png', slice_dwi)
            cv2.imwrite(os.path.join(path_t2, idx_num) + '.png', slice_t2)
            cv2.imwrite(os.path.join(path_label, idx_num) + '.png', labels)


def check_3_tumor():
    path = '/home/amax/Public/data/postate/picai_decomp'

    image_list = os.listdir(path)

    n = 0
    for image_name in image_list:
        path_2 = path + '/' + image_name + '/' + image_name + '_4.nii.gz'
        if os.path.exists(path_2):
            # print(path_2)
            n = n+1

    print(n)


if __name__ == '__main__':
    read_ni_to_image()
    # check_3_tumor()
