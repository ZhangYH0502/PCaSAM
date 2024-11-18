import torch
from torch.utils.data import DataLoader
import os
from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
import albumentations as A
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def model_input_MRI(batch):
    images_b0 = batch['image_adc'].float()
    images_b1 = batch['image_dwi'].float()
    images_b2 = batch['image_t2'].float()
    
    masks = batch['label']

    model_input = torch.cat((images_b0, images_b1, images_b2), dim=1)

    return model_input, masks


def MRIDatamodule(
        data_root,
        image_size=(1024, 1024),
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        mode='train',
):

    dataset = MRI_Dataset(data_root=data_root, image_size=image_size, mode=mode)

    output_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return output_dataloader


class MRI_Dataset(Dataset):
    def __init__(self, data_root, image_size, mode):

        self.mode = mode
        self.image_size = image_size

        self.modality = ['adc', 'dwi', 't2']
        
        if self.mode == 'train':
            self.data_root = os.path.join(data_root, 'train')
            self.data_list = os.listdir(os.path.join(self.data_root, self.modality[0]))
        else:
            self.data_root = os.path.join(data_root, 'test')
            self.data_list = os.listdir(os.path.join(self.data_root, self.modality[0]))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        image_id = self.data_list[idx]

        image_b0 = cv2.imread(os.path.join(self.data_root, self.modality[0], image_id), cv2.IMREAD_GRAYSCALE)
        image_b0 = cv2.resize(image_b0, self.image_size)
        image_b0 = self.tensor_and_normalization(image_b0)

        image_b1 = cv2.imread(os.path.join(self.data_root, self.modality[1], image_id), cv2.IMREAD_GRAYSCALE)
        image_b1 = cv2.resize(image_b1, self.image_size)
        image_b1 = self.tensor_and_normalization(image_b1)

        image_b2 = cv2.imread(os.path.join(self.data_root, self.modality[2], image_id), cv2.IMREAD_GRAYSCALE)
        image_b2 = cv2.resize(image_b2, self.image_size)
        image_b2 = self.tensor_and_normalization(image_b2)

        # label = cv2.imread(os.path.join(self.data_root, 'label', image_id), cv2.IMREAD_GRAYSCALE)
        # label = label / 100
        # label = cv2.resize(label, self.image_size)
        # label = np.array(label)
        # label = np.round(label)
        # label[label < 0] = 0
        # label[label > 2] = 2

        label_tumor = cv2.imread(os.path.join(self.data_root, 'label', image_id), cv2.IMREAD_GRAYSCALE)
        label_tumor = cv2.resize(label_tumor, self.image_size, interpolation=cv2.INTER_NEAREST)
        label_tumor = np.array(label_tumor)
        label_tumor[label_tumor > 0] = 1

        label = torch.Tensor(label_tumor)

        return {
            'image_adc': image_b0,
            'image_dwi': image_b1,
            'image_t2': image_b2,
            'image_id': image_id,
            'label': label,
        }
    
    def tensor_and_normalization(self, x):
        x = torch.Tensor(np.array(x))
        if torch.max(x) > 0:
            x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))

        return x.unsqueeze(0)


class MRI_Dataset_Aug(Dataset):
    def __init__(self, data_root, image_size, mode):

        self.mode = mode
        self.image_size = image_size

        self.modality = ['adc', 'dwi', 't2']

        if self.mode == 'train':
            self.data_root = os.path.join(data_root, 'train')
            self.data_list = os.listdir(os.path.join(self.data_root, self.modality[0]))
        else:
            self.data_root = os.path.join(data_root, 'test')
            self.data_list = os.listdir(os.path.join(self.data_root, self.modality[0]))

        self.train_transform = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Rotate((-30, 30), p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.Normalize(mean=(0.5), std=(0.5), max_pixel_value=255.0, p=1.0),
        ])

        self.test_transform = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(mean=(0.5), std=(0.5), max_pixel_value=255.0, p=1.0),
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        image_id = self.data_list[idx]

        image_b0 = np.array(cv2.imread(os.path.join(self.data_root, self.modality[0], image_id), cv2.IMREAD_GRAYSCALE))
        image_b1 = np.array(cv2.imread(os.path.join(self.data_root, self.modality[1], image_id), cv2.IMREAD_GRAYSCALE))
        image_b2 = np.array(cv2.imread(os.path.join(self.data_root, self.modality[2], image_id), cv2.IMREAD_GRAYSCALE))

        images = np.concatenate((np.expand_dims(image_b0, 2), np.expand_dims(image_b1, 2), np.expand_dims(image_b2, 2)),
                                axis=2)

        label = np.array(cv2.imread(os.path.join(self.data_root, 'label', image_id), cv2.IMREAD_GRAYSCALE))
        label = label / 100
        label_tumor = np.array(cv2.imread(os.path.join(self.data_root, 'tumor_label', image_id), cv2.IMREAD_GRAYSCALE))
        label[label_tumor > 0] = 3

        label[label != 3] = 0
        label[label == 3] = 1

        if self.mode == 'train':
            transformed = self.train_transform(image=images, mask=label)
        else:
            transformed = self.test_transform(image=images, mask=label)

        transformed_image = transformed['image']
        transformed_label = transformed['mask']

        return {
            'image_adc': torch.Tensor(transformed_image[:, :, 0]).unsqueeze(0),
            'image_dwi': torch.Tensor(transformed_image[:, :, 1]).unsqueeze(0),
            'image_t2': torch.Tensor(transformed_image[:, :, 2]).unsqueeze(0),
            'image_id': image_id,
            'label': torch.Tensor(transformed_label),
        }


def visualize(image_b0, image_b1, image_b2, mask,
              original_image_b0, original_image_b1, original_image_b2, original_mask):
    fontsize = 18

    f, ax = plt.subplots(4, 2, figsize=(8, 8))

    ax[0, 0].imshow(original_image_b0)
    ax[0, 0].set_title('Original image b0', fontsize=fontsize)

    ax[1, 0].imshow(original_image_b1)
    ax[1, 0].set_title('Original image b1', fontsize=fontsize)

    ax[2, 0].imshow(original_image_b2)
    ax[2, 0].set_title('Original image b2', fontsize=fontsize)

    ax[3, 0].imshow(original_mask)
    ax[3, 0].set_title('Original mask', fontsize=fontsize)

    ax[0, 1].imshow(image_b0)
    ax[0, 1].set_title('Transformed image b0', fontsize=fontsize)

    ax[1, 1].imshow(image_b1)
    ax[1, 1].set_title('Transformed image b1', fontsize=fontsize)

    ax[2, 1].imshow(image_b2)
    ax[2, 1].set_title('Transformed image b2', fontsize=fontsize)

    ax[3, 1].imshow(mask)
    ax[3, 1].set_title('Transformed mask', fontsize=fontsize)

    plt.show()


if __name__ == '__main__':
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.5), std=(0.5), max_pixel_value=255.0, p=1.0),
    ])

    image_b0 = np.array(cv2.imread('/home/amax/Public/data/postate/postate158_decomp/images_total/train/adc/021_9.png', cv2.IMREAD_GRAYSCALE))
    image_b1 = np.array(cv2.imread('/home/amax/Public/data/postate/postate158_decomp/images_total/train/dwi/021_9.png', cv2.IMREAD_GRAYSCALE))
    image_b2 = np.array(cv2.imread('/home/amax/Public/data/postate/postate158_decomp/images_total/train/t2/021_9.png', cv2.IMREAD_GRAYSCALE))

    images = np.concatenate((np.expand_dims(image_b0, 2), np.expand_dims(image_b1, 2), np.expand_dims(image_b2, 2)), axis=2)

    label = np.array(cv2.imread('/home/amax/Public/data/postate/postate158_decomp/images_total/train/label/021_9.png', cv2.IMREAD_GRAYSCALE))
    label = label / 100
    label_tumor = np.array(cv2.imread('/home/amax/Public/data/postate/postate158_decomp/images_total/train/tumor_label/021_9.png', cv2.IMREAD_GRAYSCALE))
    label[label_tumor > 0] = 3

    transformed = train_transform(image=images, mask=label)

    transformed_image = transformed['image']
    transformed_label = transformed['mask']

    print(np.max(images))
    print(np.max(transformed_image))

    visualize(transformed_image[:, :, 0], transformed_image[:, :, 1], transformed_image[:, :, 2], transformed_label,
              images[:, :, 0], images[:, :, 1], images[:, :, 2], label)
