import numpy as np
import random
import torch
import torch.nn.functional as F
import os
from glob import glob
from typing import Tuple
from copy import deepcopy
import torchvision.transforms as transforms

class RandomHFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, sample):
        image, masks, shape =  sample['image'], sample['masks'],sample['shape']

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image,dims=[2])
            masks = torch.flip(masks,dims=[2])
            # points = deepcopy(points).to(torch.float)
            # bboxs = deepcopy(bboxs).to(torch.float)
            # points[:, 0] = shape[-1] - points[:, 0]
            # bboxs[:, 0] = shape[-1] - bboxs[:, 2] bug???
            # bboxs[:, 0] = shape[-1] - bboxs[:, 2] - bboxs[:, 0]

        return {
            "image": image,
            "masks": masks,
            # "points": points,
            # "bboxs": bboxs,
            "shape": shape
        }

class ResizeLongestSide(object):
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length
    
    def apply_image(self, image: torch.Tensor, original_size: Tuple[int, ...]) -> torch.Tensor:
        target_size = self.get_preprocess_shape(original_size[0], original_size[1], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_boxes(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_coords(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords
    
    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __call__(self, sample):
        image, masks, shape =  sample['image'], sample['masks'],sample['shape']

        image = self.apply_image(image.unsqueeze(0), shape).squeeze(0)
        masks = self.apply_image(masks.unsqueeze(1), shape).squeeze(1)
        # points = self.apply_coords(points, shape)
        # bboxs = self.apply_boxes(bboxs, shape)

        return {
            "image": image,
            "masks": masks,
            # "points": points,
            # "bboxs": bboxs,
            "shape": shape
        }

class Normalize_and_Pad(object):
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length
        self.transform = transforms.Normalize(
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375]
        )

    def __call__(self, sample):
        image, masks, shape =  sample['image'], sample['masks'],sample['shape']

        h, w = image.shape[-2:]
        image = self.transform(image)

        padh = self.target_length - h
        padw = self.target_length - w

        image = F.pad(image.unsqueeze(0), (0, padw, 0, padh), value=0).squeeze(0)
        masks = F.pad(masks.unsqueeze(1), (0, padw, 0, padh), value=0).squeeze(1)

        return {
            "image": image,
            "masks": masks,
            # "points": points,
            # "bboxs": bboxs,
            "shape": shape
        }






    