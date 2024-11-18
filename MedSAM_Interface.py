# -*- coding: utf-8 -*-
# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join

import torch
import torch.nn.functional as F
import torch.nn as nn
from segment_anything import sam_model_registry
from skimage import io, transform
import argparse


# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


class MedSamInterface(nn.Module):
    def __init__(self,
        checkpoint="work_dir/MedSAM/medsam_vit_b.pth",
        device="cuda:0",
    ):
        super().__init__()
        self.medsam_model = sam_model_registry["vit_b"](checkpoint=checkpoint)
        self.medsam_model = self.medsam_model.to(device)
        self.medsam_model.eval()

    @torch.no_grad()
    def forward(self, image, box_torch):  # x: (1, 3, 1024, 1024)

        image_embedding = self.medsam_model.image_encoder(image)  # (1, 256, 64, 64)

        sparse_embeddings, dense_embeddings = self.medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = self.medsam_model.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        # low_res_pred1 = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        # medsam_seg = (low_res_pred1 > 0.5).astype(np.uint8)

        return low_res_pred


class MedSam(nn.Module):
    def __init__(self,
        checkpoint="work_dir/MedSAM/medsam_vit_b.pth",
        device="cuda:0",
    ):
        super().__init__()
        self.medsam_model = sam_model_registry["vit_b"](checkpoint=checkpoint)
        self.medsam_model = self.medsam_model.to(device)
        self.medsam_model.eval()

    def forward(self, image, box_1024): # x: (1, 3, 1024, 1024)
        with torch.no_grad():
            image_embedding = self.medsam_model.image_encoder(image)  # (1, 256, 64, 64)
        medsam_seg, medsam_logits = self.medsam_inference(image_embedding, box_1024)
        return medsam_seg, medsam_logits

    @torch.no_grad()
    def medsam_inference(self, img_embed, box_1024):
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = self.medsam_model.mask_decoder(
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=self.medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        low_res_pred1 = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        medsam_seg = (low_res_pred1 > 0.5).astype(np.uint8)

        return medsam_seg, low_res_pred


if __name__ == '__main__':
    # %% load model and image
    parser = argparse.ArgumentParser(
        description="run inference on testing set based on MedSAM"
    )
    parser.add_argument(
        "-i",
        "--data_path",
        type=str,
        default="assets/021_9.png",
        help="path to the data folder",
    )
    parser.add_argument(
        "-o",
        "--seg_path",
        type=str,
        default="assets/",
        help="path to the segmentation folder",
    )
    parser.add_argument(
        "--box",
        type=list,
        default=[98, 106, 143, 146], # [120, 106, 167, 146], #[98, 106, 143, 146], #[95, 255, 190, 350],
        help="bounding box of the segmentation target",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device",
    )

    args = parser.parse_args()

    medsam_model = MedSam()

    img_np = io.imread(args.data_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    H, W, _ = img_3c.shape
    # %% image preprocessing
    img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    # normalize to [0, 1], (H, W, 3)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
    # convert the shape to (3, H, W)
    img_1024_tensor = (torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(args.device))

    box_np = np.array([args.box])
    # transfer box_np t0 1024x1024 scale
    box_1024 = box_np / np.array([W, H, W, H]) * 1024

    medsam_seg, medsam_logits = medsam_model(img_1024_tensor, box_1024)

    io.imsave(
        join(args.seg_path, "seg_" + os.path.basename(args.data_path)),
        medsam_seg * 200,
        check_contrast=False,
    )

    # %% visualize results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_3c)
    show_box(box_np[0], ax[0])
    ax[0].set_title("Input Image and Bounding Box")
    ax[1].imshow(img_3c)
    show_mask(medsam_seg, ax[1])
    show_box(box_np[0], ax[1])
    ax[1].set_title("MedSAM Segmentation")
    plt.show()