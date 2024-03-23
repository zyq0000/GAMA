import torch
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
import argparse
import json
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,  default="path-to-images")
    parser.add_argument('--output_dir', type=str, default="path-to-outputs")
    parser.add_argument('--img_type', type=str, default="vit-base", help='type of image features')
    parser.add_argument('--dataset', type=str, default="dataset-name")
    args = parser.parse_args()
    return args


def extract_features(img_type, input_images):
    config = resolve_data_config({}, model=vit_model)
    transform = create_transform(**config)
    with torch.no_grad():
        inputs = []
        for image in input_images:
            img = Image.open(image).convert("RGB")
            inputs.append(transform(img).unsqueeze(0))
        inputs = torch.cat(inputs).to("cuda:0")
        feature = vit_model.forward_features(inputs)
    return feature


def find_all_file(base: str, is_full_name=True):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if not f.endswith(".jpg") and not f.endswith(".png") and not f.endswith(".jpeg"):
                continue
            if is_full_name:
                fullname = os.path.join(root, f)
                yield fullname
            else:
                yield f


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset


if __name__ == '__main__':
    args = parse_args()
    print("args", args)

    if not os.path.exists(os.path.join(args.output_dir, args.dataset, args.img_type)):
        os.makedirs(os.path.join(args.output_dir, args.dataset, args.img_type))
    all_images = list(find_all_file(args.data_root, is_full_name=True))

    tmp = None
    name_map = {}
    print(len(all_images))
    vit_model = timm.create_model("vit_base_patch16_384", pretrained=True, num_classes=0).to("cuda:0")
    vit_model.eval()
    idx = 0
    step = 300
    total_num = len(all_images)
    for idx in tqdm(range(0, total_num + step, step)):
        if idx >= total_num:
            break
        batch = all_images[idx:min(idx + step, total_num)]
        feature = extract_features(args.img_type, batch).detach().cpu()
        for i, image in enumerate(batch):
            file_name = image.split("/")[-1].split(".")[0] + ".npy"
            np.save(os.path.join(args.output_dir, args.dataset, args.img_type, file_name), feature[i].unsqueeze(dim=0))
            idx += 1
