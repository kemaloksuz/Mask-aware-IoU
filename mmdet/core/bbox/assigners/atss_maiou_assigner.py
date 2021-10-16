import torch

from ..builder import BBOX_ASSIGNERS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as patch
import random

@BBOX_ASSIGNERS.register_module()
class ATSSmaIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self, topk):
        self.topk = topk

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               gt_masks=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute maiou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding maiou for the these candidates, and compute the
           mean and std, set mean + std as the maiou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as postive
        6. limit the positive sample's center in gt


        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute maiou between all bbox and gt
        maious, ious, MOB = self.maiou_calculator(gt_bboxes, bboxes, gt_masks)
        overlaps = maious.transpose(0, 1)
        
        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            if end_idx - start_idx < self.topk:
                _, topk_idxs_per_level = distances_per_level.topk(
                (end_idx - start_idx) , dim=0, largest=False)
            else:
                _, topk_idxs_per_level = distances_per_level.topk(
                self.topk, dim=0, largest=False)

            #_, topk_idxs_per_level = distances_per_level.topk(
            #    self.topk, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def maiou_calculator(self, bboxes1, bboxes2, gt_masks, eps=1e-6):
        """Calculate maIoU between two set of bboxes and one set of mask.
        Args:
            bboxes1 (Tensor): shape (m, 4)
            bboxes2 (Tensor): shape (n, 4)
            gt_masks (Tensor): shape (m, 4) 

        Returns:
            maious(Tensor): shape (m, n) 
            ious(Tensor): shape (m, n) 
        """

        # Compute IoUs
        rows = bboxes1.size(0)
        cols = bboxes2.size(0)

        if rows * cols == 0:
            return bboxes1.new_zeros(rows, cols), bboxes1.new_zeros(rows, cols), None

        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [rows, cols, 2]
        intersection = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])

        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        union = area1[:, None] + area2 - intersection

        eps = union.new_tensor([eps])
        union = torch.max(union, eps)
        ious = intersection / union

        with torch.no_grad():
            # For efficiency only consider IoU>0 for maIoU computation
            larger_ind = ious > 0.
            overlap = bboxes1.new_zeros(rows, cols)
            if not torch.is_tensor(gt_masks):
                all_gt_masks = gt_masks.to_tensor(torch.bool, bboxes1.get_device())
            else:
                all_gt_masks = gt_masks.type(torch.cuda.BoolTensor)
            gt_number, image_h, image_w = all_gt_masks.size()

            # Compute integral image for all ground truth masks (Line 2 of Alg.1 in the paper)
            integral_images = self.integral_image_compute(all_gt_masks, gt_number, image_h, image_w).type(
                torch.cuda.FloatTensor)

            # MOB Ratio
            MOB_ratio = integral_images[:, -1, -1] / (area1 + eps)

            # For each ground truth compute maIoU
            for i in range(gt_number):
                all_boxes = torch.round(bboxes2[larger_ind[i]].clone()).type(torch.cuda.IntTensor)
                all_boxes = torch.clamp(all_boxes, min=0)
                all_boxes[:, 2] = torch.clamp(all_boxes[:, 2], max=image_w)
                all_boxes[:, 3] = torch.clamp(all_boxes[:, 3], max=image_h)
                # Compute mask-aware intersection (Eq. 3 in the paper)
                overlap[i, larger_ind[i]] = self.integral_image_fetch(integral_images[i], all_boxes)/MOB_ratio[i]
            # Normaling mask-aware intersection by union yields maIoU (Eq. 5)
            maious = overlap / union

        return maious, ious, MOB_ratio

    def integral_image_compute(self, masks, gt_number, h, w):
        integral_images = [None] * gt_number
        pad_row = torch.zeros([gt_number, 1, w]).type(torch.cuda.BoolTensor)
        pad_col = torch.zeros([gt_number, h + 1, 1]).type(torch.cuda.BoolTensor)
        integral_images = torch.cumsum(
            torch.cumsum(torch.cat([pad_col, torch.cat([pad_row, masks], dim=1)], dim=2), dim=1), dim=2)
        return integral_images

    def integral_image_fetch(self, mask, bboxes):
        TLx = bboxes[:, 0].tolist()
        TLy = bboxes[:, 1].tolist()
        BRx = bboxes[:, 2].tolist()
        BRy = bboxes[:, 3].tolist()
        area = mask[BRy, BRx] + mask[TLy, TLx] - mask[TLy, BRx] - mask[BRy, TLx]
        return area