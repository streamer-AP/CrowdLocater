import argparse
import json
import math
import os

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from coco import CocoDetection
import utils
from utils import get_local_rank, get_local_size, is_main_process
from models import build_model
from models.utils import module2model
from tqdm import tqdm
import cv2
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def inference_transform():
    return A.Compose([
        A.Resize(768,1280),
        A.Normalize(),
    ])

def draw_dmap(dmap):
    dmap = dmap[0].cpu().numpy()
    dmap = dmap / dmap.max()
    dmap = dmap * 255
    dmap = dmap.astype(np.uint8)
    dmap = cv2.applyColorMap(dmap, cv2.COLORMAP_JET)
    return dmap
def draw_points(img,points):
    h,w=img.shape[:2]
    r=h//100
    for point in points:
        cv2.circle(img,(int(point[0]),int(point[1])),r,(0,0,255),r//2)
    return img
class CountingDateSet(CocoDetection):
    def __init__(self, root, annFile, transforms=None, max_len=5000, cache_mode=False, local_rank=0, local_size=1):
        super().__init__(root, annFile, transform=None, target_transform=None,
                         transforms=None, cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self.alb_transforms = transforms
        self.to_tensor = ToTensorV2()
        self.max_len = max_len

    def __getitem__(self, index):

        image, target,img_path = super().__getitem_withpath__(index)
        img_id = self.ids[index]
        w, h = image.size
        image = np.array(image)


        data = self.alb_transforms(image=image)
        image = data["image"]
        labels = {}

        labels["wh"] = torch.as_tensor([w, h], dtype=torch.long)
        labels["id"] = torch.as_tensor(int(img_id), dtype=torch.long)
        image = self.to_tensor(image=image)["image"]
        labels["points"] = torch.zeros((self.max_len, 2), dtype=torch.float32)

        # 把 image pad成64的倍数
        h1, w1 = image.shape[-2], image.shape[-1]
        padsize = 64
        h2 = math.ceil(h1 / padsize) * padsize
        w2 = math.ceil(w1 / padsize) * padsize
        h_pad = h2 - h1
        w_pad = w2 - w1
        image_pad = F.pad(image, pad=(0, w_pad, 0, h_pad))
        labels["w1h1"] = torch.as_tensor([w1, h1], dtype=torch.long)
        labels["img_path"]=img_path

        return image_pad, labels


@torch.no_grad()
def inference(model, data_loader, args):
    model.eval()
    result = {}
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(args.gpu)
            assert inputs.shape[0] == 1
            if args.distributed:
                pred_pts,pred_maps = model.module.forward_points(inputs,threshold=0.9)
            else:
                pred_pts,pred_maps = model.forward_points(inputs)
            for i in range(len(pred_pts)):
                w, h = labels["wh"][i][0].item(), labels["wh"][i][1].item()
                w1, h1 = labels["w1h1"][i][0].item(), labels["w1h1"][i][1].item()
                result[str(labels["img_path"][0])]=[]
                for pt in pred_pts[i]:
                    x,y= pt
                    y, x = y * h / h1, x * w / w1
                    result[str(labels["img_path"][0])].append([x, y])
            if args.draw:
                img_prefix=args.Dataset.val.img_prefix if args.mode=="val" else args.Dataset.test.img_prefix
                new_name="_".join(str(labels["img_path"][0]).split(".")[0].split("/"))

                for pred_map,pt in zip(pred_maps,pred_pts):
                    img=cv2.imread(os.path.join(img_prefix,str(labels["img_path"][0]).split(".")[0]+".jpg"))
                    img = draw_points(img,result[str(labels["img_path"][0])])
                    cv2.imwrite(os.path.join(args.draw_dir,new_name+".jpg"),img)
                    dmap=draw_dmap(pred_map)
                    cv2.imwrite(os.path.join(args.draw_dir,new_name+"_dmap.jpg"),dmap)
    return result


def main(args, ckpt_path):
    utils.init_distributed_mode(args)
    utils.set_randomseed(42 + utils.get_rank())

    # initilize the model
    model = model_without_ddp = build_model(args.Model)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = module2model(ckpt)
    model_dict = model.state_dict()
    load_param_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(load_param_dict)
    model_without_ddp.load_state_dict(model_dict)
    model.cuda().eval()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # build the dataset and dataloader
    if args.mode == "val":
        prefix=args.Dataset.val.img_prefix
        ann_file=args.Dataset.val.ann_file

    elif args.mode == "test":
        prefix=args.Dataset.test.img_prefix
        ann_file=args.Dataset.test.ann_file

    else:
        prefix=args.Dataset.train.img_prefix
        ann_file=args.Dataset.train.ann_file
    dataset_test = CountingDateSet(prefix,
                                    ann_file,
                                    max_len=5000,
                                    transforms=inference_transform(),
                                    cache_mode=False,
                                    local_rank=get_local_rank(),
                                    local_size=get_local_size())

    sampler_test = DistributedSampler(
        dataset_test, shuffle=False) if args.distributed else None
    loader_val = DataLoader(dataset_test,
                            batch_size=1,
                            sampler=sampler_test,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)
    results = inference(model, loader_val, args)
    for result in results:
        path=os.path.join(args.output_dir, f"{result}.txt")
        path_dir=os.path.dirname(path)
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        with open(path,"w") as f:
            for pt in results[result]:
                f.write(f"{pt[0]},{pt[1]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DenseMap Head ")
    parser.add_argument("--config", default="fidtm.json")
    parser.add_argument("--mode", default="test")
    parser.add_argument("--draw_dir", default="locater/vis")
    parser.add_argument("--output_dir", default="locater/results")

    parser.add_argument("--vis", action="store_true")
    parser.add_argument(
        "--ckpt",
        default="/home/xinyan/hrcrowd/locater_github/weight.pth")
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            configs = json.load(f)
        cfg = edict(configs)
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        if args.vis:
            os.makedirs(args.draw_dir, exist_ok=True)
    cfg.draw_dir=args.draw_dir
    cfg.output_dir=args.output_dir
    cfg.mode=args.mode
    cfg.draw=args.vis
    main(cfg, args.ckpt)
