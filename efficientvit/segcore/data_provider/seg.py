import sys

import random
from PIL import Image

import os
from copy import deepcopy
from skimage import io
import numpy as np
import torch

import cv2

from torch.utils.data import Dataset, DataLoader
from efficientvit.apps.data_provider import DataProvider
import torchvision.transforms as transforms
from efficientvit.segcore.data_provider.utils import RandomHFlip, ResizeLongestSide, Normalize_and_Pad
from torch.utils.data.distributed import DistributedSampler
import json
from pycocotools import mask as mask_utils
from efficientvit.samcore.data_provider.utils import (
    SAMDistributedSampler,
)
import zipfile

from efficientvit.apps import setup
__all__ = ["SegDataProvider"]




class OnlineDataset(Dataset):
    def __init__(self, root, train=True, num_masks=8, transform=None,test=False):
        self.transform = transform
        self.train = train
        self.test=test
        self.num_masks = num_masks
        # print(f"self.num_masks={self.num_masks}")
        # self.root = root
        self.root=root
        self.images=[]
        self.masks=[]
                        # print(os.path.join(root_path,file))
                        # self.masks.append(os.path.join(root_path,file.replace("img","seg")))
        for root2, dir,files in os.walk(self.root):
            if files==[]:
                continue
            root_path=os.path.join(self.root,root2)
            # print(root_path)
            root_path=root2
            for file in files:
                if file.endswith(".npy") and "img" in file:
                    if file.replace("img","seg") not in files:continue
                    self.images.append(os.path.join(root_path,file))
                    # print(os.path.join(root_path,file))
                    self.masks.append(os.path.join(root_path,file.replace("img","seg")))
        # print(f"self.images={self.images}")
        prop=1.
        #proportion
        print('prop:')
        print(prop)
        val_prop=0.1
        if self.train:
            self.images=self.images[:-int(val_prop*len(self.images))]
            self.masks=self.masks[:-int(val_prop*len(self.masks))]
        else:
            self.images=self.images[-int(val_prop*len(self.images)):]
            self.masks=self.masks[-int(val_prop*len(self.masks)):]
        ls=random.sample(range(0,len(self.images)),int(prop*len(self.images)))
        random.shuffle(ls)
        self.images=[self.images[i] for i in ls]
        self.masks = [self.masks[i] for i in ls]
        print(f"len self.images:{len(self.images)}")
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):

        im=np.load(self.images[idx])
        im=np.expand_dims(im,axis=0)
        im2=np.concatenate((im,im,im))
        im = torch.tensor(im2, dtype=torch.float16)
        mask=np.load(self.masks[idx])
        mask=np.expand_dims(mask,axis=0)
        mask = torch.tensor(mask, dtype=torch.float16)
        #masks = masks.sigmoid()

        sample = {
            "image": im,
            "masks": mask,
            "shape": torch.tensor(im.shape[-2:])
        }

        # if self.transform:
        #     sample = self.transform(sample)

        return sample


class SegDataProvider(DataProvider):
    name = "med_seg"

    def __init__(
        self,
        root: str,
        sub_epochs_per_epoch: int,
        num_masks: int,
        train_batch_size: int,
        test_batch_size: int,
        valid_size: int or float or None = None,
        n_worker=8,
        image_size: int = 1024,
        num_replicas: int or None = None,
        rank: int or None = None,
        train_ratio: float or None = None,
        drop_last: bool = False,
    ):
        self.root = root
        self.num_masks = num_masks
        self.sub_epochs_per_epoch = sub_epochs_per_epoch

        super().__init__(
            train_batch_size,
            test_batch_size,
            valid_size,
            n_worker,
            image_size,
            num_replicas,
            rank,
            train_ratio,
            drop_last,
        )

    def build_train_transform(self):
        train_transforms = [
            RandomHFlip(),
            ResizeLongestSide(target_length=self.image_size[0]),
            Normalize_and_Pad(target_length=self.image_size[0]),
        ]

        return transforms.Compose(train_transforms)

    def build_valid_transform(self):
        valid_transforms = [
            ResizeLongestSide(target_length=self.image_size[0]),
            Normalize_and_Pad(target_length=self.image_size[0]),
        ]

        return transforms.Compose(valid_transforms)

    def build_datasets(self) -> tuple[any, any, any]:
        train_transform = self.build_train_transform()
        valid_transform = self.build_valid_transform()

        train_dataset = OnlineDataset(root=self.root, train=True, num_masks=2, transform=train_transform)
        val_dataset = OnlineDataset(root=self.root, train=False, num_masks=2, transform=valid_transform)

        test_dataset = None

        return train_dataset, val_dataset, test_dataset

    def build_dataloader(self, dataset: any or None, batch_size: int, n_worker: int, drop_last: bool, train: bool):
        if dataset is None:
            return None
        if train:
            sampler = SAMDistributedSampler(dataset, sub_epochs_per_epoch=self.sub_epochs_per_epoch)
            dataloader = DataLoader(dataset, batch_size, sampler=sampler, drop_last=True, num_workers=n_worker,pin_memory=True)
            return dataloader
        else:
            sampler = DistributedSampler(dataset, shuffle=False)
            dataloader = DataLoader(dataset, batch_size, sampler=sampler, drop_last=False, num_workers=n_worker,pin_memory=True)
            return dataloader

    def set_epoch_and_sub_epoch(self, epoch: int, sub_epoch: int) -> None:
        if isinstance(self.train.sampler, SAMDistributedSampler):
            self.train.sampler.set_epoch_and_sub_epoch(epoch, sub_epoch)


if __name__ == "__main__":
    # transform = transforms.Compose([RandomHFlip(), ResizeLongestSide(target_length=512), Normalize_and_Pad(target_length=512)])
    # # dataset = OnlineDataset(root="/lustre/fsw/nvresearch/zhuoyangz/data/sam")
    # dataset = OnlineDataset(root="/lustre/fsw/nvresearch/zhuoyangz/data/sam", transform=transform)
    # # for i in range(100):
    #     # dataset[i]
    # sample = dataset[0]
    # img = sample["image"]
    # masks = sample["masks"]
    # print(img.shape)
    # io.imsave(f'/lustre/fsw/nvresearch/zhuoyangz/efficientvit-dev/efficientvit/samlightcore/data_provider/img.jpg', img.permute(1,2,0).numpy().astype(np.uint8))
    # for i in range(len(masks)):
    #     io.imsave(f'/lustre/fsw/nvresearch/zhuoyangz/efficientvit-dev/efficientvit/samlightcore/data_provider/mask_{i}.png', (masks[i]*255).numpy().astype(np.uint8))

    setup.setup_dist_env(None)
    # config = setup.setup_exp_config("/lustre/fsw/nvresearch/zhuoyangz/efficientvit-dev/efficientvit/samlightcore/data_provider/try.yaml", recursive=True, opt_args=None)
    config = setup.setup_exp_config("/lustre/fsw/nvresearch/zhuoyangz/efficientvit-dev/configs/samlight/l0.yaml", recursive=True, opt_args=None)
    data_provider = setup.setup_data_provider(config, [SAMLightDataProvider], is_distributed=True)

    # from efficientvit.sam_model_zoo import create_sam_model
    # efficientvit_sam = create_sam_model("l0", True, "/lustre/fsw/nvresearch/zhuoyangz/efficientvit-dev/assets/checkpoints/sam/l0.pt").cuda().eval()

    # print(len(data_provider.valid))
    print(len(data_provider.train))
    # for data in data_provider.valid:
    for data in data_provider.train:
        # image, masks, bboxs, points = data['image'], data['masks'], data["bboxs"] * 2, data["points"] * 2
        image, masks, bboxs, points, shape = data['image'], data['masks'], data["bboxs"] * 2, data["points"] * 2, data["shape"]
        bboxs[..., 2] = bboxs[..., 0] + bboxs[..., 2]
        bboxs[..., 3] = bboxs[..., 1] + bboxs[..., 3]
        # print(bboxs[0,1])
        # print(points[0,1])
        # print(image.shape)
        # io.imsave('/lustre/fsw/nvresearch/zhuoyangz/efficientvit-dev/efficientvit/samlightcore/data_provider/input.jpg',image[0].permute(1,2,0).numpy().astype(np.uint8))
        # io.imsave('/lustre/fsw/nvresearch/zhuoyangz/efficientvit-dev/efficientvit/samlightcore/data_provider/gt.png',(masks[0,1]*255).numpy().astype(np.uint8))

        # # print(bboxs[0,1])
        # # input_dic = [{"image": image[0].cuda(), "boxes": bboxs[0,0].unsqueeze(0).cuda()}, 
        # #     {"image": image[0].cuda(), "boxes": bboxs[0,0].unsqueeze(0).cuda()}]
        # # input_dic = [{"image": image[0].cuda(), "boxes": bboxs[0].cuda()}, 
        # #     {"image": image[0].cuda(), "boxes": bboxs[0].cuda()}]
        # print(points.shape)
        # input_dic = [{"image": image[0].cuda(), "point_coords": points[0].unsqueeze(1).cuda(), "point_labels": torch.tensor([[1],[1]]).cuda()}, 
        #     {"image": image[0].cuda(), "point_coords": points[0].unsqueeze(1).cuda(), "point_labels": torch.tensor([[1],[1]]).cuda()}]
        # outputs = efficientvit_sam(input_dic, False)
        # print(outputs.shape)
        # io.imsave('/lustre/fsw/nvresearch/zhuoyangz/efficientvit-dev/efficientvit/samlightcore/data_provider/pred.png',((outputs[0,1]>0.)*255).cpu().detach().numpy().astype(np.uint8))
        break