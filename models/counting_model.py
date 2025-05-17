import os

import torch
from mmcv.cnn import get_model_complexity_info
from torch import nn

from .backbone import build_backbone
from .counting_head import build_counting_head
from .utils import module2model
from torch.amp import autocast
from torch.nn import functional as F
import numpy as np
def topklocalmax(pmap):
    cnt = torch.round(pmap.sum()).long().detach().cpu().item()
    H, W = pmap.shape[-2], pmap.shape[-1]
    pmap_unflod = F.unfold(pmap, kernel_size=3, padding=1)
    pred_max = pmap_unflod.max(dim=1, keepdim=True)[1]
    pred_max = (pred_max == 3 ** 2 // 2).reshape(1, 1, H, W).detach().cpu()
    max_indices = torch.arange(H*W)[pred_max.reshape(-1)] 
    cnt = min(cnt, len(max_indices))
    pred_pts = []
    if cnt == 0:
        return pred_pts

    pred_cnt = pmap_unflod[:, :, max_indices]
    pred_cnt = pred_cnt.sum(dim=[0, 1])
    val, indices = torch.topk(pred_cnt, k=cnt, dim=0, largest=True)

    coord_h = torch.arange(H).reshape(1, 1, H, 1).repeat(1, 1, 1, W).float().cuda()
    coord_w = torch.arange(W).reshape(1, 1, 1, W).repeat(1, 1, H, 1).float().cuda()
    coord_h_unflod = F.unfold(coord_h, kernel_size=3, padding=1) 
    coord_w_unflod = F.unfold(coord_w, kernel_size=3, padding=1)
    pred_coord_weight = F.normalize(pmap_unflod, p=1, dim=1)
    coord_h_pred = (coord_h_unflod * pred_coord_weight).sum(dim=1, keepdim=True).reshape(H*W)
    coord_w_pred = (coord_w_unflod * pred_coord_weight).sum(dim=1, keepdim=True).reshape(H*W)
    coord_h = coord_h_pred.unsqueeze(1)
    coord_w = coord_w_pred.unsqueeze(1)
    coord = torch.cat((coord_h, coord_w), dim=1) 
    coord = coord[max_indices]
    coord = coord[indices].cpu().numpy()

    for pt in coord:
        y_coord, x_coord = pt
        y_coord, x_coord = float(4 * y_coord + 1.5), float(4 * x_coord + 1.5)
        pred_pts.append([x_coord, y_coord])
    return pred_pts

class SingleScaleEncoderDecoderCounting(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.backbone = build_backbone(args.backbone)
        self.decoder_layers = build_counting_head(args.counting_head)
        self.stride=4
    
    @autocast("cuda")
    def forward(self, x, ext_info=None):

        z = self.backbone(x)
        out_dict = self.decoder_layers(z)
        return out_dict

    def get_model_complexity(self, input_shape):
        flops, params = get_model_complexity_info(self, input_shape)
        return flops, params

    @torch.no_grad()
    def forward_points(self, x, threshold=0.8,loc_kernel_size=3):
        assert loc_kernel_size%2==1
        assert x.shape[0]==1
        z = self.backbone(x)
        out_dict = self.decoder_layers(z)
        predict_counting_map=out_dict["predict_counting_map"].detach().float()
        pred_points=self._map_to_points(predict_counting_map,threshold=threshold,loc_kernel_size=loc_kernel_size,device=x.device)

        return [pred_points],predict_counting_map
    @torch.no_grad()
    def _map_to_points(self, predict_counting_map, threshold=0.8,loc_kernel_size=3,device="cuda"):
        loc_padding=loc_kernel_size//2
        kernel=torch.ones(1,1,loc_kernel_size,loc_kernel_size).to(device).float()

        threshold=0.5
        low_resolution_map=F.interpolate(F.relu(predict_counting_map),scale_factor=1)
        H,W=low_resolution_map.shape[-2],low_resolution_map.shape[-1]

        unfolded_map=F.unfold(low_resolution_map,kernel_size=loc_kernel_size,padding=loc_padding)
        unfolded_max_idx=unfolded_map.max(dim=1,keepdim=True)[1]
        unfolded_max_mask=(unfolded_max_idx==loc_kernel_size**2//2).reshape(1,1,H,W)

        predict_cnt=F.conv2d(low_resolution_map,kernel,padding=loc_padding)
        predict_filter=(predict_cnt>threshold).float()
        predict_filter=predict_filter*unfolded_max_mask
        predict_filter=predict_filter.detach().cpu().numpy().astype(bool).reshape(H,W)

        pred_coord_weight=F.normalize(unfolded_map,p=1,dim=1)
        
        coord_h=torch.arange(H).reshape(-1,1).repeat(1,W).to(device).float()
        coord_w=torch.arange(W).reshape(1,-1).repeat(H,1).to(device).float()
        coord_h=coord_h.unsqueeze(0).unsqueeze(0)
        coord_w=coord_w.unsqueeze(0).unsqueeze(0)
        unfolded_coord_h=F.unfold(coord_h,kernel_size=loc_kernel_size,padding=loc_padding)
        pred_coord_h=(unfolded_coord_h*pred_coord_weight).sum(dim=1,keepdim=True).reshape(H,W).detach().cpu().numpy()
        unfolded_coord_w=F.unfold(coord_w,kernel_size=loc_kernel_size,padding=loc_padding)
        pred_coord_w=(unfolded_coord_w*pred_coord_weight).sum(dim=1,keepdim=True).reshape(H,W).detach().cpu().numpy()
        coord_h=pred_coord_h[predict_filter].reshape(-1,1)
        coord_w=pred_coord_w[predict_filter].reshape(-1,1)
        coord=np.concatenate([coord_w,coord_h],axis=1)

        pred_points=[[self.stride*coord_w+0.5,self.stride*coord_h+0.5] for coord_w,coord_h in coord]
        return pred_points

def build_counting_model(args):
    model = SingleScaleEncoderDecoderCounting(args)

    if os.path.exists(args.ckpt):
        print("load param from", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        state_dict = module2model(ckpt['model'])
        model.load_state_dict(state_dict, strict=False)

    return model
