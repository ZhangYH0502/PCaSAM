import os
import numpy as np
import cv2


if __name__ == '__main__':
    path = '/home/amax/Public/MedSAM-main/results/postate158.OurModel.CrossFuseWithoutBox'

    path_in = path + '/' + 'result_masks'
    path_out = path + '/' + 'result_masks_processed'

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    file_list = os.listdir(path_in)

    for name in file_list:
        image = cv2.imread(path_in + '/' + name, cv2.IMREAD_GRAYSCALE)
        image[image > 0] = 255

        image = cv2.erode(image, kernel=np.ones((3, 3), dtype=np.uint8), iterations=2)
        image = cv2.dilate(image, kernel=np.ones((3, 3), dtype=np.uint8), iterations=2)

        cv2.imwrite(path_out + '/' + name, image)
