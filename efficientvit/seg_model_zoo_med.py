# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

from efficientvit.models.efficientvit import (
    EfficientViTSeg,
    efficientvit_seg_b0,
    efficientvit_seg_b1,
    efficientvit_seg_b2,
    efficientvit_seg_b3,
    efficientvit_seg_l1,
    efficientvit_seg_l2,
    efficientvit_sam_l1,
)
from efficientvit.models.nn.norm import set_norm_eps
from efficientvit.models.utils import load_state_dict_from_file

__all__ = ["create_seg_model"]


REGISTERED_SEG_MODEL: dict[str, dict[str, str]] = {
    "cityscapes": {
        "b0": "assets/checkpoints/seg/cityscapes/b0.pt",
        "b1": "assets/checkpoints/seg/cityscapes/b1.pt",
        "b2": "assets/checkpoints/seg/cityscapes/b2.pt",
        "b3": "assets/checkpoints/seg/cityscapes/b3.pt",
        ################################################
        "l1": "assets/checkpoints/seg/cityscapes/l1.pt",
        "l2": "assets/checkpoints/seg/cityscapes/l2.pt",
    },
    "ade20k": {
        "b1": "assets/checkpoints/seg/ade20k/b1.pt",
        "b2": "assets/checkpoints/seg/ade20k/b2.pt",
        "b3": "assets/checkpoints/seg/ade20k/b3.pt",
        ################################################
        "l1": "assets/checkpoints/seg/ade20k/l1.pt",
        "l2": "assets/checkpoints/seg/ade20k/l2.pt",
    },
}
def load_pretrain(model,weight_url):
    weight_url = weight_url
    if weight_url is None:
        raise ValueError(f"Do not find the pretrained weight.")
    else:
        weight = load_state_dict_from_file(weight_url)
        model.load_state_dict(weight)
    return model

def create_seg_model(
    name: str, dataset: str, pretrained_backbone=True,pretrained_head=False, weight_url_backbone: str or None = None,weight_url_head: str or None = None, **kwargs
) -> EfficientViTSeg:
    model_dict = {
        "b0": efficientvit_seg_b0,
        "b1": efficientvit_seg_b1,
        "b2": efficientvit_seg_b2,
        "b3": efficientvit_seg_b3,
        #########################
        "l1": efficientvit_seg_l1,
        "l2": efficientvit_seg_l2,
    }
    sam_dict={
        "l1": efficientvit_sam_l1,
    }

    model_id = name.split("-")[0]
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](dataset=dataset, **kwargs)
        model_head=model_dict[model_id](dataset=dataset, **kwargs)
        model_backbone=sam_dict[model_id](**kwargs)

    if model_id in ["l1", "l2"]:
        set_norm_eps(model, 1e-7)
    

    # if weight_url is None:
    #     raise ValueError(f"Do not find the pretrained weight.")
    # else:
    if pretrained_backbone:
        weight = load_state_dict_from_file(weight_url_backbone)
        model_backbone.load_state_dict(weight)
        model.backbone=model_backbone.image_encoder.backbone
    

    return model

