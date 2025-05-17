from .timm_model import build_backbone as build_backbone_timm

def build_backbone(args):
    return build_backbone_timm(args)