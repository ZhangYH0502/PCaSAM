import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
import random
import cv2

from networks.unet import U_Net
from torchvision.models import resnet18
from networks.region_to_location import find_connected_components as fic
from networks.fusion import *
from segment_anything import sam_model_registry


class Customize_Model_TTA(nn.Module):
    def __init__(self):
        super(Customize_Model_TTA, self).__init__()

        self.medsam_model = sam_model_registry["vit_b"](checkpoint="work_dir/MedSAM/medsam_vit_b.pth")
        self.medsam_model = self.medsam_model.to("cuda:0")
        self.medsam_model.eval()

        self.feature_fusion_for_coarse_seg = CrossAttentionFusion()
        checkpoint = torch.load('networks/feature_fuse.pt')
        self.feature_fusion_for_coarse_seg.load_state_dict(checkpoint['state_dict'])
        self.feature_fusion_for_coarse_seg = self.feature_fusion_for_coarse_seg.to(device="cuda:0")
        self.feature_fusion_for_coarse_seg.eval()

        self.feature_fusion_for_prompt_seg = CrossAttentionFusion()
        checkpoint = torch.load('networks/feature_fuse.pt')
        self.feature_fusion_for_prompt_seg.load_state_dict(checkpoint['state_dict'])
        self.feature_fusion_for_prompt_seg = self.feature_fusion_for_prompt_seg.to(device="cuda:0")
        self.feature_fusion_for_prompt_seg.eval()

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, y):

        image_embedd, image_embedd0, image_embedd1, image_embedd2 = self.image_embed(x)

        image_embedding_fused_0 = self.feature_fusion_for_coarse_seg(image_embedd, image_embedd0, image_embedd1, image_embedd2)
        image_embedding_fused_1 = self.feature_fusion_for_prompt_seg(image_embedd, image_embedd0, image_embedd1, image_embedd2)

        coarse_masks = self.coarse_seg(image_embedding_fused_0)

        # encoding path
        boxes, offset_width, offset_height = self.prompt_gen(masks=coarse_masks, first=True)
        pred_masks = self.sam_seg(image_embedding_fused_1, boxes)

        for i in range(3):
            boxes, offset_width, offset_height = self.prompt_gen(masks=pred_masks, offset_width=offset_width, offset_height=offset_height, first=False)
            pred_masks = self.sam_seg(image_embedding_fused_1, boxes)

        return pred_masks

    @torch.no_grad()
    def coarse_seg(self, image_embedding):
        pred_masks = self.sam_seg(image_embedding, boxes=None)
        # pred_masks[pred_masks > 0.5] = 1
        # pred_masks[pred_masks <= 0.5] = 0
        return pred_masks

    @torch.no_grad()
    def prompt_gen(self, masks, offset_width=None, offset_height=None, first=True):

        y = masks[0, :, :]
        y[y > 0.5] = 1
        y[y <= 0.5] = 0

        boxes = torch.zeros((1, 1, 4), device=masks.device)

        nonzero_indices = torch.nonzero(y)

        min_x = torch.min(nonzero_indices[:, 1])
        min_y = torch.min(nonzero_indices[:, 0])
        max_x = torch.max(nonzero_indices[:, 1])
        max_y = torch.max(nonzero_indices[:, 0])

        if first:
            offset_width = (max_x - min_x + 1) * 0.3
            offset_height = (max_y - min_y + 1) * 0.3
        else:
            offset_width = offset_width * 0.7
            offset_height = offset_height * 0.7

        boxes[0, 0, 0] = min_x / 256 * 1024 - offset_width
        boxes[0, 0, 1] = min_y / 256 * 1024 - offset_height
        boxes[0, 0, 2] = max_x / 256 * 1024 + offset_width
        boxes[0, 0, 3] = max_y / 256 * 1024 + offset_height

        boxes = torch.clip(boxes, min=0, max=1024)

        del y

        return boxes, offset_width, offset_height

    @torch.no_grad()
    def image_embed(self, image):

        image0 = image[:, 0:1, :, :]
        image0 = F.interpolate(image0, size=(1024, 1024), mode="bilinear")
        image0 = torch.cat((image0, image0, image0), dim=1)
        image_embedding0 = self.medsam_model.image_encoder(image0)  # (1, 256, 64, 64)

        image1 = image[:, 1:2, :, :]
        image1 = F.interpolate(image1, size=(1024, 1024), mode="bilinear")
        image1 = torch.cat((image1, image1, image1), dim=1)
        image_embedding1 = self.medsam_model.image_encoder(image1)  # (1, 256, 64, 64)

        image2 = image[:, 2:3, :, :]
        image2 = F.interpolate(image2, size=(1024, 1024), mode="bilinear")
        image2 = torch.cat((image2, image2, image2), dim=1)
        image_embedding2 = self.medsam_model.image_encoder(image2)  # (1, 256, 64, 64)

        image_all = torch.cat((image0[:, 0:1, :, :], image1[:, 0:1, :, :], image2[:, 0:1, :, :]), dim=1)
        image_embedding = self.medsam_model.image_encoder(image_all)  # (1, 256, 64, 64)

        del image0, image1, image2, image_all

        return image_embedding, image_embedding0, image_embedding1, image_embedding2

    def sam_seg(self, image_embedding, boxes):  # x: (1, 3, 1024, 1024)

        sparse_embeddings, dense_embeddings = self.medsam_model.prompt_encoder(
            points=None,
            boxes=boxes,
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
        # low_res_pred = F.interpolate(low_res_pred, size=(256, 512), mode="bilinear")

        # low_res_pred = low_res_pred.squeeze()
        # low_res_pred[low_res_pred > 0.5] = 1
        # low_res_pred[low_res_pred <= 0.5] = 0

        # low_res_pred1 = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        # medsam_seg = (low_res_pred1 > 0.5).astype(np.uint8)

        return low_res_pred


class Customize_Model_statistic(nn.Module):
    def __init__(self):
        super(Customize_Model_statistic, self).__init__()

        self.medsam_model = sam_model_registry["vit_b"](checkpoint="work_dir/MedSAM/medsam_vit_b.pth")
        self.medsam_model = self.medsam_model.to("cuda:0")
        self.medsam_model.eval()

        self.feature_fusion_for_coarse_seg = CrossAttentionFusion()
        checkpoint = torch.load('networks/feature_fusion_for_coarse_seg.pt')
        self.feature_fusion_for_coarse_seg.load_state_dict(checkpoint['state_dict'])
        self.feature_fusion_for_coarse_seg = self.feature_fusion_for_coarse_seg.to(device="cuda:0")
        self.feature_fusion_for_coarse_seg.eval()

        self.feature_fusion_for_prompt_seg = CrossAttentionFusion()
        checkpoint = torch.load('networks/feature_fusion_for_prompt_seg.pt')
        self.feature_fusion_for_prompt_seg.load_state_dict(checkpoint['state_dict'])
        self.feature_fusion_for_prompt_seg = self.feature_fusion_for_prompt_seg.to(device="cuda:0")
        self.feature_fusion_for_prompt_seg.eval()

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, y):

        image_embedd, image_embedd0, image_embedd1, image_embedd2 = self.image_embed(x)

        image_embedding_fused_0 = self.feature_fusion_for_coarse_seg(image_embedd, image_embedd0, image_embedd1,
                                                                     image_embedd2)
        image_embedding_fused_1 = self.feature_fusion_for_prompt_seg(image_embedd, image_embedd0, image_embedd1,
                                                                     image_embedd2)

        pred_masks = self.coarse_seg(image_embedding_fused_0)
        boxes = self.prompt_gen(pred_masks)

        final_masks = torch.zeros(size=y.shape).cuda()
        final_logits = []

        for box in boxes:
            pred_logits_box, pred_masks_box = self.sam_seg(image_embedding_fused_1, box)

            for i in range(1):
                box_2 = self.prompt_gen_2(pred_masks_box)
                pred_logits_box, pred_masks_box = self.sam_seg(image_embedding_fused_1, box_2)


            # tmp_masks = pred_masks_box + y
            # if torch.max(tmp_masks) < 2:
            #     continue
            # else:
            #     final_masks[pred_masks_box == 1] = 1
            #     final_logits.append(pred_logits_box)

            final_masks[pred_masks_box == 1] = 1
            final_logits.append(pred_logits_box)

        if len(final_logits) == 0:
            final_logits = final_masks[:, None, :, :]
        else:
            final_logits = torch.cat(final_logits, dim=1)
        final_logits, _ = torch.max(final_logits, dim=1, keepdim=True)

        # for i in range(3):
        #     offset = 10
        #     boxes = self.prompt_gen(pred_masks[:, 0, :, :], offset)
        #     pred_masks = self.sam_seg(image_embedding_fused_1, boxes)

        return final_logits, final_masks

    @torch.no_grad()
    def coarse_seg(self, image_embedding):
        _, pred_masks = self.sam_seg(image_embedding, boxes=None, thr=0.6)
        pred_masks = pred_masks[0, :, :]

        pred_masks_1 = pred_masks.data.cpu().numpy()
        pred_masks_1 = cv2.erode(pred_masks_1, kernel=np.ones((3, 3), dtype=np.uint8), iterations=2)
        pred_masks_1 = cv2.dilate(pred_masks_1, kernel=np.ones((3, 3), dtype=np.uint8), iterations=2)

        pred_masks_1 = np.array(pred_masks_1)
        pred_masks_1 = torch.Tensor(pred_masks_1)

        return pred_masks_1.cuda()

    @torch.no_grad()
    def prompt_gen(self, masks):

        y = masks.data.cpu().numpy().astype(np.uint8)

        num_labels, labels, states, centroids = cv2.connectedComponentsWithStats(y, connectivity=8)

        boxes_total = []

        for i in range(1, num_labels):

            yi = np.array(labels == i)

            nonzero_indices = torch.nonzero(torch.Tensor(yi))

            min_x = torch.min(nonzero_indices[:, 1])
            min_y = torch.min(nonzero_indices[:, 0])
            max_x = torch.max(nonzero_indices[:, 1])
            max_y = torch.max(nonzero_indices[:, 0])

            offset_x = torch.round((max_x - min_x) * 0.15)
            offset_y = torch.round((max_y - min_y) * 0.15)

            boxes = torch.zeros((1, 1, 4), device=masks.device)
            boxes[0, 0, 0] = min_x / 256 * 1024 - offset_x
            boxes[0, 0, 1] = min_y / 256 * 1024 - offset_y
            boxes[0, 0, 2] = max_x / 256 * 1024 + offset_x
            boxes[0, 0, 3] = max_y / 256 * 1024 + offset_y

            boxes = torch.clip(boxes, min=0, max=1023)

            boxes_total.append(boxes)

        del y

        return boxes_total

    @torch.no_grad()
    def prompt_gen_2(self, masks):
        masks = masks[0, :, :]

        nonzero_indices = torch.nonzero(torch.Tensor(masks))

        min_x = torch.min(nonzero_indices[:, 1])
        min_y = torch.min(nonzero_indices[:, 0])
        max_x = torch.max(nonzero_indices[:, 1])
        max_y = torch.max(nonzero_indices[:, 0])

        offset_x = torch.round((max_x - min_x) * 0.15)
        offset_y = torch.round((max_y - min_y) * 0.15)

        boxes = torch.zeros((1, 1, 4), device=masks.device)
        boxes[0, 0, 0] = min_x / 256 * 1024 - offset_x
        boxes[0, 0, 1] = min_y / 256 * 1024 - offset_y
        boxes[0, 0, 2] = max_x / 256 * 1024 + offset_x
        boxes[0, 0, 3] = max_y / 256 * 1024 + offset_y

        boxes = torch.clip(boxes, min=0, max=1023)

        return boxes

    @torch.no_grad()
    def image_embed(self, image):

        image0 = image[:, 0:1, :, :]
        image0 = F.interpolate(image0, size=(1024, 1024), mode="bilinear")
        image0 = torch.cat((image0, image0, image0), dim=1)
        image_embedding0 = self.medsam_model.image_encoder(image0)  # (1, 256, 64, 64)

        image1 = image[:, 1:2, :, :]
        image1 = F.interpolate(image1, size=(1024, 1024), mode="bilinear")
        image1 = torch.cat((image1, image1, image1), dim=1)
        image_embedding1 = self.medsam_model.image_encoder(image1)  # (1, 256, 64, 64)

        image2 = image[:, 2:3, :, :]
        image2 = F.interpolate(image2, size=(1024, 1024), mode="bilinear")
        image2 = torch.cat((image2, image2, image2), dim=1)
        image_embedding2 = self.medsam_model.image_encoder(image2)  # (1, 256, 64, 64)

        image_all = torch.cat((image0[:, 0:1, :, :], image1[:, 0:1, :, :], image2[:, 0:1, :, :]), dim=1)
        image_embedding = self.medsam_model.image_encoder(image_all)  # (1, 256, 64, 64)

        del image0, image1, image2, image_all

        return image_embedding, image_embedding0, image_embedding1, image_embedding2

    def sam_seg(self, image_embedding, boxes, thr=0.5):  # x: (1, 3, 1024, 1024)

        sparse_embeddings, dense_embeddings = self.medsam_model.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        low_res_logits, _ = self.medsam_model.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_logits = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        # low_res_pred = F.interpolate(low_res_pred, size=(256, 512), mode="bilinear")

        low_res_pred = low_res_logits[:, 0, :, :]
        low_res_pred[low_res_pred > thr] = 1
        low_res_pred[low_res_pred <= thr] = 0

        # low_res_pred1 = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        # medsam_seg = (low_res_pred1 > 0.5).astype(np.uint8)

        return low_res_logits, low_res_pred


class Customize_Model_JointTraining(nn.Module):
    def __init__(self):
        super(Customize_Model_JointTraining, self).__init__()

        self.medsam_model = sam_model_registry["vit_b"](checkpoint="work_dir/MedSAM/medsam_vit_b.pth")
        self.medsam_model = self.medsam_model.to("cuda:0")
        self.medsam_model.image_encoder.eval()
        self.medsam_model.mask_decoder.eval()

        self.feature_fusion_for_coarse_seg = CrossAttentionFusion()
        checkpoint = torch.load('networks/feature_fusion_for_coarse_seg.pt')
        self.feature_fusion_for_coarse_seg.load_state_dict(checkpoint['state_dict'])
        self.feature_fusion_for_coarse_seg = self.feature_fusion_for_coarse_seg.to(device="cuda:0")
        # self.feature_fusion_for_coarse_seg.eval()

        self.feature_fusion_for_prompt_seg = CrossAttentionFusion()
        checkpoint = torch.load('networks/feature_fusion_for_prompt_seg.pt')
        self.feature_fusion_for_prompt_seg.load_state_dict(checkpoint['state_dict'])
        self.feature_fusion_for_prompt_seg = self.feature_fusion_for_prompt_seg.to(device="cuda:0")
        self.feature_fusion_for_prompt_seg.eval()

        for p in self.parameters():
            p.requires_grad = False

        for p in self.medsam_model.prompt_encoder.parameters():
            p.requires_grad = True

        for p in self.feature_fusion_for_coarse_seg.parameters():
            p.requires_grad = True

    def forward(self, x, y, val=True):

        image_embedd, image_embedd0, image_embedd1, image_embedd2 = self.image_embed(x)

        image_embedding_fused_0 = self.feature_fusion_for_coarse_seg(image_embedd, image_embedd0, image_embedd1,
                                                                     image_embedd2)
        image_embedding_fused_1 = self.feature_fusion_for_prompt_seg(image_embedd, image_embedd0, image_embedd1,
                                                                     image_embedd2)

        pred_masks = self.coarse_seg(image_embedding_fused_0)

        final_logits, final_masks = self.sam_seg(image_embedding_fused_1, masks=pred_masks)

        return final_logits, final_masks, y

    @torch.no_grad()
    def coarse_seg(self, image_embedding):
        _, pred_masks = self.sam_seg(image_embedding, boxes=None, thr=0.5)
        pred_masks = pred_masks.unsqueeze(1)

        return pred_masks

    @torch.no_grad()
    def image_embed(self, image):

        image0 = image[:, 0:1, :, :]
        image0 = F.interpolate(image0, size=(1024, 1024), mode="bilinear")
        image0 = torch.cat((image0, image0, image0), dim=1)
        image_embedding0 = self.medsam_model.image_encoder(image0)  # (1, 256, 64, 64)

        image1 = image[:, 1:2, :, :]
        image1 = F.interpolate(image1, size=(1024, 1024), mode="bilinear")
        image1 = torch.cat((image1, image1, image1), dim=1)
        image_embedding1 = self.medsam_model.image_encoder(image1)  # (1, 256, 64, 64)

        image2 = image[:, 2:3, :, :]
        image2 = F.interpolate(image2, size=(1024, 1024), mode="bilinear")
        image2 = torch.cat((image2, image2, image2), dim=1)
        image_embedding2 = self.medsam_model.image_encoder(image2)  # (1, 256, 64, 64)

        image_all = torch.cat((image0[:, 0:1, :, :], image1[:, 0:1, :, :], image2[:, 0:1, :, :]), dim=1)
        image_embedding = self.medsam_model.image_encoder(image_all)  # (1, 256, 64, 64)

        del image0, image1, image2, image_all

        return image_embedding, image_embedding0, image_embedding1, image_embedding2

    def sam_seg(self, image_embedding, boxes=None, masks=None, thr=0.5):  # x: (1, 3, 1024, 1024)

        sparse_embeddings, dense_embeddings = self.medsam_model.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=masks,
        )
        low_res_logits, _ = self.medsam_model.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_logits = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        # low_res_pred = F.interpolate(low_res_pred, size=(256, 512), mode="bilinear")

        low_res_pred = low_res_logits[:, 0, :, :].clone()
        low_res_pred[low_res_pred > thr] = 1
        low_res_pred[low_res_pred <= thr] = 0

        # low_res_pred1 = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        # medsam_seg = (low_res_pred1 > 0.5).astype(np.uint8)

        return low_res_logits, low_res_pred


class Customize_Model_TrainingBox(nn.Module):
    def __init__(self):
        super(Customize_Model_TrainingBox, self).__init__()

        self.medsam_model = sam_model_registry["vit_b"](checkpoint="work_dir/MedSAM/medsam_vit_b.pth")
        self.medsam_model = self.medsam_model.to("cuda:0")
        self.medsam_model.eval()

        self.feature_fusion_for_prompt_seg = CrossAttentionFusion()
        checkpoint = torch.load('networks/feature_fusion_for_prompt_seg.pt')
        self.feature_fusion_for_prompt_seg.load_state_dict(checkpoint['state_dict'])
        self.feature_fusion_for_prompt_seg = self.feature_fusion_for_prompt_seg.to(device="cuda:0")
        self.feature_fusion_for_prompt_seg.eval()

        self.box_generater = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, 5),
            nn.Sigmoid(),
        )
        self.box_generater = self.box_generater.to(device="cuda:0")

        for p in self.parameters():
            p.requires_grad = False

        for p in self.box_generater.parameters():
            p.requires_grad = True

    def forward(self, x, y, val=True):

        image_embedd, image_embedd0, image_embedd1, image_embedd2 = self.image_embed(x)

        image_embedding_fused = self.feature_fusion_for_prompt_seg(image_embedd, image_embedd0, image_embedd1,
                                                                     image_embedd2)

        boxes = self.box_generater(x)
        # boxes = boxes.view(x.shape[0], 1, 5)

        boxes_leftx = boxes[:, 0] * 1024
        boxes_lefty = boxes[:, 1] * 1024
        boxes_width = boxes[:, 2] * 1024
        boxes_height = boxes[:, 3] * 1024

        boxes_bottomx = boxes_leftx + boxes_width
        boxes_bottomy = boxes_lefty + boxes_height

        boxes_coor = torch.stack((boxes_leftx, boxes_lefty, boxes_bottomx, boxes_bottomy), dim=1).unsqueeze(1)
        boxes_coor = torch.clip(boxes_coor, min=0, max=1023)
        boxes_score = boxes[:, -1]

        gt_box = self.prompt_init(y)
        gt_score = torch.ones(size=boxes_score.shape, device=x.device)

        coor_loss = torch.mean((boxes_coor - gt_box) ** 2)
        score_loss = F.binary_cross_entropy(input=boxes_score, target=gt_score, reduction='mean')

        pred_logits_box, pred_masks_box = self.sam_seg(image_embedding_fused, boxes_coor)

        return pred_logits_box, pred_masks_box, y, coor_loss+score_loss

    def prompt_init(self, y):
        B = y.shape[0]

        boxes = torch.zeros((B, 1, 4), device=y.device)

        for idx in range(B):
            nonzero_indices = torch.nonzero(y[idx, :, :])

            min_x = torch.min(nonzero_indices[:, 1])
            min_y = torch.min(nonzero_indices[:, 0])
            max_x = torch.max(nonzero_indices[:, 1])
            max_y = torch.max(nonzero_indices[:, 0])

            offset_x = torch.round((max_x - min_x) * 0.1)
            offset_y = torch.round((max_y - min_y) * 0.1)

            boxes[idx, 0, 0] = min_x / 256 * 1024 - offset_x
            boxes[idx, 0, 1] = min_y / 256 * 1024 - offset_y
            boxes[idx, 0, 2] = max_x / 256 * 1024 + offset_x
            boxes[idx, 0, 3] = max_y / 256 * 1024 + offset_y

        boxes = torch.clip(boxes, min=0, max=1023)

        return boxes

    def image_embed(self, image):
        image0 = image[:, 0:1, :, :]
        image0 = F.interpolate(image0, size=(1024, 1024), mode="bilinear")
        image0 = torch.cat((image0, image0, image0), dim=1)
        image_embedding0 = self.medsam_model.image_encoder(image0)  # (1, 256, 64, 64)

        image1 = image[:, 1:2, :, :]
        image1 = F.interpolate(image1, size=(1024, 1024), mode="bilinear")
        image1 = torch.cat((image1, image1, image1), dim=1)
        image_embedding1 = self.medsam_model.image_encoder(image1)  # (1, 256, 64, 64)

        image2 = image[:, 2:3, :, :]
        image2 = F.interpolate(image2, size=(1024, 1024), mode="bilinear")
        image2 = torch.cat((image2, image2, image2), dim=1)
        image_embedding2 = self.medsam_model.image_encoder(image2)  # (1, 256, 64, 64)

        image_all = torch.cat((image0[:, 0:1, :, :], image1[:, 0:1, :, :], image2[:, 0:1, :, :]), dim=1)
        image_embedding = self.medsam_model.image_encoder(image_all)  # (1, 256, 64, 64)

        del image0, image1, image2, image_all

        return image_embedding, image_embedding0, image_embedding1, image_embedding2

    def sam_seg(self, image_embedding, boxes, thr=0.5):  # x: (1, 3, 1024, 1024)

        sparse_embeddings, dense_embeddings = self.medsam_model.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        low_res_logits, _ = self.medsam_model.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_logits = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        # low_res_pred = F.interpolate(low_res_pred, size=(256, 512), mode="bilinear")

        low_res_pred = low_res_logits[:, 0, :, :].clone()
        low_res_pred[low_res_pred > thr] = 1
        low_res_pred[low_res_pred <= thr] = 0

        # low_res_pred1 = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        # medsam_seg = (low_res_pred1 > 0.5).astype(np.uint8)

        return low_res_logits, low_res_pred


class Customize_Model(nn.Module):
    def __init__(self):
        super(Customize_Model, self).__init__()

        self.medsam_model = sam_model_registry["vit_b"](checkpoint="work_dir/MedSAM/medsam_vit_b.pth")
        self.medsam_model = self.medsam_model.to("cuda:0")
        self.medsam_model.eval()

        # self.coarse_seg = U_Net(img_ch=3, output_ch=2)
        # checkpoint = torch.load('networks/unet.pt')
        # self.coarse_seg.load_state_dict(checkpoint['state_dict'])
        # self.coarse_seg = self.coarse_seg.to(device="cuda:1")
        # self.coarse_seg.eval()

        for p in self.medsam_model.parameters():
            p.requires_grad = False

        # self.coor_fine = CoordinateRefine()
        # self.coor_fine = self.coor_fine.to("cuda:1")

        self.feature_fusion = CrossAttentionFusion()
        # checkpoint = torch.load('networks/feature_fuse.pt')
        # self.feature_fusion.load_state_dict(checkpoint['state_dict'])
        self.feature_fusion = self.feature_fusion.to(device="cuda:0")
        # self.feature_fusion.eval()

        for p in self.feature_fusion.parameters():
            p.requires_grad = True

        for p in self.medsam_model.prompt_encoder.mask_downscaling.parameters():
            p.requires_grad = True

    def forward(self, x, y, val=False):
        x_new, masks_new = self.reshape_data_single_region(x, y)

        image_embedding, image_embedding0, image_embedding1, image_embedding2 = self.image_embed(x_new)

        del x, x_new

        image_embedding_fused = self.feature_fusion(image_embedding, image_embedding0, image_embedding1, image_embedding2)

        # encoding path
        boxes = self.prompt_init(masks_new)
        # masks_lack = self.mask_init(masks_new)

        pred_logits, pred_masks = self.sam_seg(image_embedding_fused, boxes=boxes)

        if val:
            pred_logits, _ = torch.max(pred_logits, dim=0, keepdim=True)
            pred_masks, _ = torch.max(pred_masks, dim=0, keepdim=True)
            masks_new = y

        return pred_logits, pred_masks, masks_new

    def reshape_data_single_region(self, x, y):
        x_new = []
        y_new = []

        B = y.shape[0]
        y_np = y.data.cpu().numpy().astype(np.uint8)

        for i in range(B):
            yi = y_np[i, :, :]
            num_labels, labels, states, centroids = cv2.connectedComponentsWithStats(yi, connectivity=8)
            for j in range(1, num_labels):
                x_new.append(x[i, :, :, :])

                yj = np.array(labels == j)
                yj = torch.Tensor(yj)
                y_new.append(yj)

        x_new = torch.stack(x_new, dim=0)
        y_new = torch.stack(y_new, dim=0)

        del x, y

        return x_new, y_new.cuda()

    def mask_init(self, y):
        B = y.shape[0]

        masks = []

        for idx in range(B):
            yb = y[idx, :, :]
            nonzero_indices = torch.nonzero(yb)
            nonzero_indices = nonzero_indices[torch.randperm(nonzero_indices.shape[0]), :]
            selected_length = int(nonzero_indices.shape[0] * 0.5) + 2
            nonzero_indices = nonzero_indices[0:selected_length, :]
            yb[nonzero_indices[:, 0], nonzero_indices[:, 1]] = 0
            masks.append(yb)

        masks = torch.stack(masks, dim=0)
        return masks

    def prompt_init(self, y):
        B = y.shape[0]

        boxes = torch.zeros((B, 1, 4), device=y.device)

        for idx in range(B):
            nonzero_indices = torch.nonzero(y[idx, :, :])

            min_x = torch.min(nonzero_indices[:, 1])
            min_y = torch.min(nonzero_indices[:, 0])
            max_x = torch.max(nonzero_indices[:, 1])
            max_y = torch.max(nonzero_indices[:, 0])

            # offset_x = torch.round((max_x - min_x) * 0.15)
            # offset_y = torch.round((max_y - min_y) * 0.15)

            boxes[idx, 0, 0] = min_x / 256 * 1024
            boxes[idx, 0, 1] = min_y / 256 * 1024
            boxes[idx, 0, 2] = max_x / 256 * 1024
            boxes[idx, 0, 3] = max_y / 256 * 1024

        offsets = torch.randint_like(input=boxes, low=-10, high=20, device=y.device)
        boxes = boxes + offsets

        boxes = torch.clip(boxes, min=0, max=1023)

        del y

        return boxes

    @torch.no_grad()
    def image_embed(self, image):

        image0 = image[:, 0:1, :, :]
        image0 = F.interpolate(image0, size=(1024, 1024), mode="bilinear")
        image0 = torch.cat((image0, image0, image0), dim=1)
        image_embedding0 = self.medsam_model.image_encoder(image0)  # (1, 256, 64, 64)

        image1 = image[:, 1:2, :, :]
        image1 = F.interpolate(image1, size=(1024, 1024), mode="bilinear")
        image1 = torch.cat((image1, image1, image1), dim=1)
        image_embedding1 = self.medsam_model.image_encoder(image1)  # (1, 256, 64, 64)

        image2 = image[:, 2:3, :, :]
        image2 = F.interpolate(image2, size=(1024, 1024), mode="bilinear")
        image2 = torch.cat((image2, image2, image2), dim=1)
        image_embedding2 = self.medsam_model.image_encoder(image2)  # (1, 256, 64, 64)

        image_all = torch.cat((image0[:, 0:1, :, :], image1[:, 0:1, :, :], image2[:, 0:1, :, :]), dim=1)
        image_embedding = self.medsam_model.image_encoder(image_all)  # (1, 256, 64, 64)

        del image0, image1, image2, image

        return image_embedding, image_embedding0, image_embedding1, image_embedding2

    def sam_seg(self, image_embedding, boxes=None, masks=None):  # x: (1, 3, 1024, 1024)

        sparse_embeddings, dense_embeddings = self.medsam_model.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=masks,
        )
        low_res_logits, _ = self.medsam_model.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_logits = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        low_res_pred = low_res_logits[:, 0, :, :].clone()
        low_res_pred[low_res_pred > 0.5] = 1
        low_res_pred[low_res_pred <= 0.5] = 0

        # low_res_pred1 = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        # medsam_seg = (low_res_pred1 > 0.5).astype(np.uint8)

        return low_res_logits, low_res_pred


class SAMforCoarseSeg(nn.Module):
    def __init__(self):
        super(SAMforCoarseSeg, self).__init__()

        self.medsam_model = sam_model_registry["vit_b"](checkpoint="work_dir/MedSAM/medsam_vit_b.pth")
        self.medsam_model = self.medsam_model.to("cuda:0")
        self.medsam_model.eval()

        self.feature_fusion_q = CrossAttentionFusion()
        self.feature_fusion_k0 = CrossAttentionFusion()
        self.feature_fusion_k1 = CrossAttentionFusion()
        self.feature_fusion_k2 = CrossAttentionFusion()

        self.neck0 = nn.Sequential(
            nn.Conv2d(9216, 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

        self.neck1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

        for p in self.parameters():
            p.requires_grad = True

        for p in self.medsam_model.parameters():
            p.requires_grad = False

    def forward(self, x, y):

        outputs0, outputs1 = self.image_embed(x)

        outputs1_0 = self.neck0(outputs1[0])
        outputs1_1 = self.neck0(outputs1[1])
        outputs1_2 = self.neck0(outputs1[2])
        outputs1_3 = self.neck0(outputs1[3])

        q = self.neck1(torch.cat((outputs0[3], outputs1_3), dim=1))
        k0 = self.neck1(torch.cat((outputs0[0], outputs1_0), dim=1))
        k1 = self.neck1(torch.cat((outputs0[1], outputs1_1), dim=1))
        k2 = self.neck1(torch.cat((outputs0[2], outputs1_2), dim=1))

        image_embedding_fused_q = self.feature_fusion_q(q, k0, k1, k2)
        image_embedding_fused_k0 = self.feature_fusion_k0(k0, q, k1, k2)
        image_embedding_fused_k1 = self.feature_fusion_k1(k1, k0, q, k2)
        image_embedding_fused_k2 = self.feature_fusion_k2(k2, k0, k1, q)

        # encoding path
        # boxes = self.prompt_init(y)

        pred_logits_q, pred_masks_q = self.sam_seg(image_embedding_fused_q, boxes=None)
        pred_logits_k0, pred_masks_k0 = self.sam_seg(image_embedding_fused_k0, boxes=None)
        pred_logits_k1, pred_masks_k1 = self.sam_seg(image_embedding_fused_k1, boxes=None)
        pred_logits_k2, pred_masks_k2 = self.sam_seg(image_embedding_fused_k2, boxes=None)

        pred_masks_ass = pred_masks_k0 + pred_masks_k1 + pred_masks_k2
        pred_masks_ass[pred_masks_ass > 0] = 1

        pred_masks = pred_masks_q + pred_masks_ass
        pred_masks[pred_masks < 2] = 0
        pred_masks[pred_masks == 2] = 1

        pred_logits = [
            pred_logits_q,
            pred_logits_k0,
            pred_logits_k1,
            pred_logits_k2
        ]

        return pred_logits, pred_masks

    def prompt_init(self, y):
        B = y.shape[0]

        boxes = torch.zeros((B, 1, 4), device=y.device)

        for idx in range(B):
            nonzero_indices = torch.nonzero(y[idx, :, :])

            min_x = torch.min(nonzero_indices[:, 1])
            min_y = torch.min(nonzero_indices[:, 0])
            max_x = torch.max(nonzero_indices[:, 1])
            max_y = torch.max(nonzero_indices[:, 0])

            offset_x = torch.round((max_x - min_x) * 0.15)
            offset_y = torch.round((max_y - min_y) * 0.15)

            boxes[idx, 0, 0] = min_x / 256 * 1024 - offset_x
            boxes[idx, 0, 1] = min_y / 256 * 1024 - offset_y
            boxes[idx, 0, 2] = max_x / 256 * 1024 + offset_x
            boxes[idx, 0, 3] = max_y / 256 * 1024 + offset_y

        boxes = torch.clip(boxes, min=0, max=1023)

        del y

        return boxes

    @torch.no_grad()
    def image_embed(self, image):

        image0 = image[:, 0:1, :, :]
        image0 = F.interpolate(image0, size=(1024, 1024), mode="bilinear")
        image0 = torch.cat((image0, image0, image0), dim=1)
        image_embed00, image_embed01 = self.medsam_model.image_encoder(image0)  # (1, 256, 64, 64)

        image1 = image[:, 1:2, :, :]
        image1 = F.interpolate(image1, size=(1024, 1024), mode="bilinear")
        image1 = torch.cat((image1, image1, image1), dim=1)
        image_embed10, image_embed11 = self.medsam_model.image_encoder(image1)  # (1, 256, 64, 64)

        image2 = image[:, 2:3, :, :]
        image2 = F.interpolate(image2, size=(1024, 1024), mode="bilinear")
        image2 = torch.cat((image2, image2, image2), dim=1)
        image_embed20, image_embed21 = self.medsam_model.image_encoder(image2)  # (1, 256, 64, 64)

        image_all = torch.cat((image0[:, 0:1, :, :], image1[:, 0:1, :, :], image2[:, 0:1, :, :]), dim=1)
        image_embed_all0, image_embed_all1 = self.medsam_model.image_encoder(image_all)  # (1, 256, 64, 64)

        del image0, image1, image2, image

        outputs0 = [
            image_embed00,
            image_embed10,
            image_embed20,
            image_embed_all0,
        ]

        outputs1 = [
            image_embed01,
            image_embed11,
            image_embed21,
            image_embed_all1,
        ]

        return outputs0, outputs1

    def sam_seg(self, image_embedding, boxes):  # x: (1, 3, 1024, 1024)

        sparse_embeddings, dense_embeddings = self.medsam_model.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        low_res_logits, _ = self.medsam_model.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_logits = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        low_res_pred = low_res_logits[:, 0, :, :].clone()
        low_res_pred[low_res_pred > 0.5] = 1
        low_res_pred[low_res_pred <= 0.5] = 0

        # low_res_pred1 = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        # medsam_seg = (low_res_pred1 > 0.5).astype(np.uint8)

        return low_res_logits, low_res_pred


class SamFinetune(nn.Module):
    def __init__(self):
        super(SamFinetune, self).__init__()

        self.medsam_model = sam_model_registry["vit_b"](checkpoint="work_dir/MedSAM/medsam_vit_b.pth")
        # self.medsam_model = self.medsam_model.to("cuda:1")
        # self.medsam_model.eval()

        for p in self.parameters():
            p.requires_grad = True

        # for p in self.medsam_model.image_encoder.patch_embed.parameters():
        #     p.requires_grad = True

    def forward(self, x):
        x = F.interpolate(x, size=(1024, 1024), mode="bilinear")
        x = self.medsam_model.image_encoder(x)
        pred_masks = self.sam_seg(x)
        return pred_masks

    def sam_seg(self, image_embedding, points_torch=None, boxes_torch=None, masks_torch=None):  # x: (1, 3, 1024, 1024)

        sparse_embeddings, dense_embeddings = self.medsam_model.prompt_encoder(
            points=points_torch,
            boxes=boxes_torch,
            masks=masks_torch,
        )
        low_res_logits, _ = self.medsam_model.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        # low_res_pred = low_res_pred.squeeze()
        # low_res_pred[low_res_pred > 0.5] = 1
        # low_res_pred[low_res_pred <= 0.5] = 0

        # low_res_pred1 = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        # medsam_seg = (low_res_pred1 > 0.5).astype(np.uint8)

        return low_res_pred


class SamFinetune_last_combine(nn.Module):
    def __init__(self):
        super(SamFinetune_last_combine, self).__init__()

        self.medsam_model = sam_model_registry["vit_b"](checkpoint="work_dir/MedSAM/medsam_vit_b.pth")
        # self.medsam_model = self.medsam_model.to("cuda:1")
        # self.medsam_model.eval()

        for p in self.medsam_model.parameters():
            p.requires_grad = False

        self.last_combine = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0),
        )

        for p in self.last_combine.parameters():
            p.requires_grad = True

    def forward(self, x):
        x1 = F.interpolate(x, size=(1024, 1024), mode="bilinear")
        x1 = self.medsam_model.image_encoder(x1)
        pred_masks_1 = self.sam_seg(x1)

        x2 = F.interpolate(x[:, 0:1, :, :], size=(1024, 1024), mode="bilinear")
        x2 = torch.cat((x2, x2, x2), dim=1)
        x2 = self.medsam_model.image_encoder(x2)
        pred_masks_2 = self.sam_seg(x2)

        x3 = F.interpolate(x[:, 1:2, :, :], size=(1024, 1024), mode="bilinear")
        x3 = torch.cat((x3, x3, x3), dim=1)
        x3 = self.medsam_model.image_encoder(x3)
        pred_masks_3 = self.sam_seg(x3)

        x4 = F.interpolate(x[:, 2:3, :, :], size=(1024, 1024), mode="bilinear")
        x4 = torch.cat((x4, x4, x4), dim=1)
        x4 = self.medsam_model.image_encoder(x4)
        pred_masks_4 = self.sam_seg(x4)

        pred_masks = torch.cat((pred_masks_1, pred_masks_2, pred_masks_3, pred_masks_4), dim=1)

        x5 = F.interpolate(x, size=(256, 256), mode="bilinear")
        x5 = torch.cat((x5, pred_masks), dim=1)

        pred_masks = self.last_combine(x5)
        pred_masks = torch.sigmoid(pred_masks)

        pred_masks = F.interpolate(pred_masks, size=(512, 512), mode="bilinear")

        return pred_masks

    def sam_seg(self, image_embedding, points_torch=None, boxes_torch=None, masks_torch=None):  # x: (1, 3, 1024, 1024)

        sparse_embeddings, dense_embeddings = self.medsam_model.prompt_encoder(
            points=points_torch,
            boxes=boxes_torch,
            masks=masks_torch,
        )
        low_res_logits, _ = self.medsam_model.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        # low_res_pred = low_res_pred.squeeze()
        # low_res_pred[low_res_pred > 0.5] = 1
        # low_res_pred[low_res_pred <= 0.5] = 0

        # low_res_pred1 = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        # medsam_seg = (low_res_pred1 > 0.5).astype(np.uint8)

        return low_res_pred


class SAM_For_Test(nn.Module):
    def __init__(self):
        super(SAM_For_Test, self).__init__()

        self.medsam_model = sam_model_registry["vit_b"](checkpoint="work_dir/MedSAM/medsam_vit_b.pth")
        self.medsam_model = self.medsam_model.to("cuda:0")
        self.medsam_model.eval()

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        image_embedding = self.image_embed(x.to(device="cuda:0"))

        # encoding path
        boxes = self.prompt_init(y.to(device="cuda:0"))

        pred_masks = self.sam_seg(image_embedding, boxes=boxes)

        return pred_masks

    @torch.no_grad()
    def prompt_init(self, y):

        B = y.shape[0]

        boxes = torch.zeros((B, 1, 4), device=y.device)

        for idx in range(B):
            nonzero_indices = torch.nonzero(y[idx, :, :])

            min_x = torch.min(nonzero_indices[:, 1])
            min_y = torch.min(nonzero_indices[:, 0])
            max_x = torch.max(nonzero_indices[:, 1])
            max_y = torch.max(nonzero_indices[:, 0])

            offset_x = torch.round((max_x - min_x) * 0.15)
            offset_y = torch.round((max_y - min_y) * 0.15)

            boxes[idx, 0, 0] = min_x / 256 * 1024 - offset_x
            boxes[idx, 0, 1] = min_y / 256 * 1024 - offset_y
            boxes[idx, 0, 2] = max_x / 256 * 1024 + offset_x
            boxes[idx, 0, 3] = max_y / 256 * 1024 + offset_y

        boxes = torch.clip(boxes, min=0, max=1023)
        
        return boxes

    @torch.no_grad()
    def image_embed(self, image):
        # image = image[:, 0:1, :, :]
        # image = torch.cat((image, image, image), dim=1)
        image = F.interpolate(image, size=(1024, 1024), mode='bilinear')  # (1, 1, gt.shape)
        image_embedding = self.medsam_model.image_encoder(image)  # (1, 256, 64, 64)

        return image_embedding

    @torch.no_grad()
    def sam_seg(self, image_embedding, boxes):  # x: (1, 3, 1024, 1024)

        sparse_embeddings, dense_embeddings = self.medsam_model.prompt_encoder(
            points=None,
            boxes=boxes,
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

        # low_res_pred = low_res_pred.squeeze()
        # low_res_pred[low_res_pred > 0.5] = 1
        # low_res_pred[low_res_pred <= 0.5] = 0

        # low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)

        low_res_pred[low_res_pred > 0.5] = 1
        low_res_pred[low_res_pred <= 0.5] = 0

        return low_res_pred


class CoordinateRefine(nn.Module):
    def __init__(self):
        super(CoordinateRefine, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=2)

        self.image_embedding = nn.Conv2d(256, 128, kernel_size=(8, 8), stride=8, padding=0)

        self.final_proj = nn.Linear(in_features=128, out_features=1)

        self.offet = nn.Parameter(torch.zeros(1, 4, 128))

        self.position = PositionalEncoding()

    def forward(self, x, coor):
        B = x.shape[0]

        x_embedding = self.image_embedding(x)

        x_embedding = x_embedding.view(B, 128, -1)

        x_embedding = x_embedding.permute(0, 2, 1)

        c_embedding = self.offet.expand(x.shape[0], -1, -1) + self.position(coor[:, 0, :].long())

        seq = torch.cat((c_embedding, x_embedding), dim=1)
        seq = self.transformer_encoder(seq)
        seq = seq[:, :1, :]
        seq = self.final_proj(seq)

        seq = torch.sigmoid(seq) * 50

        seq = seq.permute(0, 2, 1)

        seq = torch.clip(seq + coor, min=0, max=223)
        seq = seq / 224 * 1024

        return seq


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

        self.position = torch.arange(0, 224).unsqueeze(1).cuda()
        self.div_term = torch.exp(torch.arange(0, 128, 2) * -(torch.log(torch.tensor(10000.0)) / 128)).cuda()
        self.pe = torch.zeros(224, 128).cuda()
        self.pe[:, 0::2] = torch.sin(self.position * self.div_term)
        self.pe[:, 1::2] = torch.cos(self.position * self.div_term)

    def forward(self, x):
        B, L = x.shape[0], x.shape[1]
        x = x.view(-1)
        pes = self.pe[x, :]
        pes = pes.view(B, L, -1)
        return pes.cuda()
