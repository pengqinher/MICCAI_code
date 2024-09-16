import random
import sys
from monai.losses.dice import *  # NOQA
import torch
from monai.losses.dice import DiceLoss
dice=DiceLoss(reduction='none',to_onehot_y=True)
import torch
import torch.nn as nn
from torchmetrics import Dice
c=nn.MSELoss()
import torch.nn.functional as F
# import wandb
from PIL import Image
from tqdm import tqdm

import torchvision.transforms as transforms
from efficientvit.segcore.data_provider.utils import RandomHFlip, ResizeLongestSide, Normalize_and_Pad

from efficientvit.apps.trainer import Trainer
from efficientvit.apps.utils import AverageMeter, get_dist_local_rank, get_dist_size, is_master, sync_tensor
from efficientvit.models.utils import list_join
from efficientvit.segcore.data_provider import SegDataProvider
from efficientvit.segcore.trainer import SegRunConfig
from efficientvit.segcore.trainer.utils import (
    compute_boundary_iou,
    compute_iou,
    loss_masks_med,
    mask_iou_batch,
    masks_sample_points,
    mask_iou
)

__all__ = ["SegTrainer"]

# def loss_fcn(gt, pred):
#     L_seg = torch_dice_coef_loss(gt, pred)
#     return L_seg


# def loss_fcn(y_true, y_pred, smooth=1.):
#     y_true_f = torch.flatten(y_true)
#     y_pred_f = torch.flatten(y_pred)
#     intersection = torch.sum(y_true_f * y_pred_f)
#     f (intersection/torch.sum(y_pred)<0.8):self.r=9
#     else:self.r=1
#     return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))

# def loss_fcn_train(y_true, y_pred, smooth=1.,r=9.):
#     y_true_f = torch.flatten(y_true)
#     y_pred_f = torch.flatten(y_pred)
#     intersection = torch.sum(y_true_f * y_pred_f)
#     return 1. - (((r+1) * intersection + smooth) / (r*torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))


#"segout"
class SegTrainer(Trainer):
    def __init__(
        self,
        path: str,
        model: nn.Module,
        data_provider: SegDataProvider,
    ) -> None:
        super().__init__(
            path=path,
            model=model,
            data_provider=data_provider,
        )
        # self.trans()
        print("changed loss fcn")
        self.best_loss=100
        self.r=100000
        # if is_master():
        #     self.wandb_log = wandb.init(project="efficientvit-Seg")
    def loss_fcn(self,y_true, y_pred, smooth=1.):
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        # if (intersection/torch.sum(y_pred)<0.8):self.r=5
        # else:self.r=1
        ratio=intersection/(torch.sum(y_true_f)+0.001)
        ratio2=(torch.sum(y_pred_f)+0.001)/(torch.sum(y_true_f)+0.001)
        return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)),ratio,ratio2
    def loss_fcn_train(self,y_true, y_pred, smooth=1.):
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        return 1. - (((self.r+1) * intersection + smooth) / (self.r*torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))
    def after_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        # gradient clip
        if self.run_config.grad_clip is not None:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.run_config.grad_clip)
        # update
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.run_config.lr_schedule_name =="platform":pass
        else: self.lr_scheduler.step()
        self.run_config.step()
        # update ema
        if self.ema is not None:
            self.ema.step(self.network, self.run_config.global_step)
    def _validate(self, model, data_loader, epoch: int, sub_epoch: int) -> dict[str, any]:
        val_loss = AverageMeter()
        val_iou = AverageMeter()
        val_ratio=AverageMeter()
        val_ratio2=AverageMeter()
        val_iou_boundary = AverageMeter()
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
            with torch.no_grad():
                with tqdm(
                    total=len(data_loader),
                    desc=f"Validate Epoch #{epoch + 1}, Sub Epoch #{sub_epoch+1}",
                    disable=not is_master(),
                    file=sys.stdout,
                ) as t:
                    for i, data in enumerate(data_loader):
                        # print("deal_max")
                        image = self.deal_max(data["image"].cuda(non_blocking=True))
                        masks = data["masks"].cuda(non_blocking=True)
                        # print
                        # image=self.deal_max(image)
                        output= self.model(image)
                        masks = masks.reshape(-1, image.shape[2], image.shape[3]).unsqueeze(1)
                        output = (
                            F.interpolate(output, size=(image.shape[2],image.shape[3]), mode="bilinear")
                            # F.interpolate(output[:, :, i], size=(image.shape[2], image.shape[3]), mode="bilinear")
                        )
                        # print(torch.mean(output))

                        output=F.sigmoid(output)
                        # loss=dice(output,masks).mean()*20
                        loss,ratio,ratio2=self.loss_fcn(output,masks)
                        # loss=c(output,masks)
                        # print(loss)
                        # if(loss<0.25):self.model.freeze=False
                        iou = compute_iou(output, masks * 255)
                        boundary_iou = compute_boundary_iou(output, masks * 255)

                        loss = sync_tensor(loss)
                        iou = sync_tensor(iou)
                        boundary_iou = sync_tensor(boundary_iou)
                        val_ratio.update(ratio, image.shape[0] * get_dist_size())
                        val_ratio2.update(ratio2, image.shape[0] * get_dist_size())
                        val_loss.update(loss, image.shape[0] * get_dist_size())
                        val_iou.update(iou, image.shape[0] * get_dist_size())
                        val_iou_boundary.update(boundary_iou, image.shape[0] * get_dist_size())

                        t.set_postfix(
                            {
                                "loss": val_loss.avg,
                                "iou": val_iou.avg,
                                "r":val_ratio.avg,
                                "r2":val_ratio2.avg,
                                "boundary_iou": val_iou_boundary.avg,
                                "bs": image.shape[0] * get_dist_size(),
                            }
                        )
                        t.update()

        if val_ratio.avg<0.7:self.r=100000.
        else: self.r=300
        return {
            "val_loss": val_loss.avg,
            "val_iou": val_iou.avg,
            "val_boundary_iou": val_iou_boundary.avg,
        }

    def validate(self, model=None, data_loader=None, epoch=0, sub_epoch=0) -> dict[str, any]:
        model = model or self.eval_network
        if data_loader is None:
            data_loader = self.data_provider.valid

        model.eval()
        return self._validate(model, data_loader, epoch, sub_epoch)

    def deal_max(self,image,im_max=2000,rgb_max=255.):
        image[image < -im_max] = -im_max
        image[image > im_max] = im_max
        image = (image - (-im_max)) / (im_max - (-im_max))
        # image=image-image.min()
        image = image*rgb_max
        # print(f"image.max={image.max()}")
        return image
    # def trans(self):
    #     train_transforms = [
    #         RandomHFlip(),
    #         ResizeLongestSide(target_length=512),
    #         Normalize_and_Pad(target_length=512),
    #     ]
    #     self.transform=transforms.Compose(train_transforms)
    def before_step(self, feed_dict: dict[str, any]) -> dict[str, any]:
        feed_dict={
            "image":self.deal_max(feed_dict["image"].cuda(non_blocking=True)),
            "masks": feed_dict["masks"].cuda(non_blocking=True),
            # "shape":feed_dict["shape"].cuda()
        }
        # feed_dict=self.transform(feed_dict)
        return feed_dict
        # image = feed_dict["image"].cuda()
        # masks = feed_dict["masks"].cuda()
        # # image=self.deal_max(image)
        # return {
        #     "image": image,
        #     "masks": masks,
        # }

    def run_step(self, feed_dict: dict[str, any]) -> dict[str, any]:
        image = feed_dict["image"]
        masks = feed_dict["masks"]
        # print(f"mask.sum={masks.sum()}")
        # print(f"max:{masks.max()}")
        # print(f"image.shape={image.shape}")
        # batched_input = []
        # for b_i in range(len(image)):
        #     dict_input = dict()
        #     dict_input["image"] = image[b_i]


        #     batched_input.append(dict_input)

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
            # print(self.amp_dtype)
            output= self.model(feed_dict["image"])

            # print(f"output.shape={output.shape}")
            # output= self.model(batched_input)
            masks = masks.reshape(-1, image.shape[2], image.shape[3]).unsqueeze(1)
            # print(masks[1,:,:,:].max())
            # print(f"mas.ks.shape={masks.shape}")
            loss_list = []
            output = (
                F.interpolate(output, size=(image.shape[2],image.shape[3]), mode="bilinear")
                # F.interpolate(output[:, :, i], size=(image.shape[2], image.shape[3]), mode="bilinear")
            )
            # output[output>255]=1.
            # output[output<=0]=0.
            output=F.sigmoid(output)
            # print(f"output.sum={output.sum()}")
            mse=c(output,masks)
            # lf=loss_fcn(output,masks)
            lf=self.loss_fcn_train(output,masks)
            # iou_loss=1.-mask_iou(output,masks)
            # print(iou_loss)
            # print(f"lf={lf}")
            # print(f"mes={mse}")
            # print(f"dloss={dloss}")
            loss=lf*100
            # loss=lf*10+iou_loss*10
            # loss=loss.mean()
        # self.scaler.scale(loss).backward()
        self.scaler.scale(loss).backward()

        return {"loss": loss, "output": output,"output_sum":output.sum()/image.shape[0]}

    def _train_one_sub_epoch(self, epoch: int, sub_epoch: int) -> dict[str, any]:
        train_loss = AverageMeter()

        with tqdm(
            total=len(self.data_provider.train),
            desc=f"Train Epoch #{epoch + 1}, Sub Epoch #{sub_epoch + 1}",
            disable=not is_master(),
            file=sys.stdout,
        ) as t:
            for i, data in enumerate(self.data_provider.train):
                feed_dict = data

                # preprocessing
                feed_dict = self.before_step(feed_dict)
                # clear gradient
                self.optimizer.zero_grad()
                # forward & backward
                output_dict = self.run_step(feed_dict)
                # update: optimizer, lr_scheduler
                self.after_step()

                loss = output_dict["loss"]
                loss = sync_tensor(loss)
                train_loss.update(loss, data["image"].shape[0] * get_dist_size())

                # if is_master():
                #     self.wandb_log.log(
                #         {
                #             "train_loss": train_loss.avg,
                #             "epoch": epoch,
                #             "sub_epoch": sub_epoch,
                #             # "output.sum":output_dict["output.sum"].item(),
                #             "learning_rate": sorted(set([group["lr"] for group in self.optimizer.param_groups]))[0],
                #         }
                #     )

                t.set_postfix(
                    {
                        "loss": train_loss.avg,
                        "bs": data["image"].shape[0] * get_dist_size(),
                        "r":self.r,
                        "lr": list_join(
                            sorted(set([group["lr"] for group in self.optimizer.param_groups])),
                            "#",
                            "%.1E",
                        ),
                        "output_sum":output_dict["output_sum"].item(),
                        "progress": self.run_config.progress,
                    }
                )
                t.update()

        return {
            "train_loss": train_loss.avg,
        }

    def train_one_sub_epoch(self, epoch: int, sub_epoch: int) -> dict[str, any]:
        self.model.train()

        self.data_provider.set_epoch_and_sub_epoch(epoch, sub_epoch)

        train_info_dict = self._train_one_sub_epoch(epoch, sub_epoch)

        return train_info_dict

    def train(self) -> None:
        val_loss_bar=0.1
        if self.run_config.lr_schedule_name =="platform":print("platform!")
        for sub_epoch in range(self.start_epoch, self.run_config.n_epochs):
            epoch = sub_epoch // self.data_provider.sub_epochs_per_epoch

            train_info_dict = self.train_one_sub_epoch(epoch, sub_epoch)

            val_info_dict = self.validate(epoch=epoch, sub_epoch=sub_epoch)

            val_loss = val_info_dict["val_loss"]
            if self.run_config.lr_schedule_name =="platform":
                print(f"step!val_loss={val_loss}")
                self.lr_scheduler.step(val_loss)
            else: pass
            if(val_loss<0.01):
                print("stop freeze!")
                self.model.freeze=False
            is_best = val_loss < self.best_loss
            self.best_loss = min(val_loss, self.best_loss)
            print(f"best_loss={self.best_loss}")
            self.save_model(
                only_state_dict=False,
                epoch=sub_epoch,
                model_name=f"checkpoint_{epoch}_{sub_epoch}.pt",
            )
            if is_best:            
                self.save_model(
                only_state_dict=False,
                epoch=sub_epoch,
                model_name=f"model_best.pt",
            )

    def prep_for_training(self, run_config: SegRunConfig, amp="fp32") -> None:
        self.run_config = run_config
        self.model = nn.parallel.DistributedDataParallel(
            self.model.cuda(),
            device_ids=[get_dist_local_rank()],
            find_unused_parameters=True,
        )

        self.run_config.global_step = 0
        self.run_config.batch_per_epoch = len(self.data_provider.train)
        assert self.run_config.batch_per_epoch > 0, "Training set is empty"

        # build optimizer
        self.optimizer, self.lr_scheduler = self.run_config.build_optimizer(self.model)

        # amp
        self.amp = amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)
