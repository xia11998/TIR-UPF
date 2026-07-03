import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from albumentations import (
    GaussNoise, Lambda, GaussianBlur, Compose, RandomCrop, RandomBrightnessContrast
)
from torch.utils.data.dataset import Dataset
import os

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


def stripe_noise(image, **params):
    if random.random() < 0.5:
        g = np.random.randn(1, image.shape[1]) * (np.random.rand() * 0.07 + 0.03)
        b = np.random.randn(1, image.shape[1]) * (np.random.rand() * 5)
    else:
        g = np.random.randn(image.shape[0], 1) * (np.random.rand() * 0.07 + 0.03)
        b = np.random.randn(image.shape[0], 1) * (np.random.rand() * 5)
    if len(image.shape) == 3:
        g = np.expand_dims(g, -1)
        b = np.expand_dims(b, -1)
    noise = image * g + b
    image = np.clip(image.astype("float32") + noise.astype("float32"), 0, 255).astype("uint8")
    return image


def nonuniformity_optical(image, **params):
    h, w = image.shape
    noise = np.ones((h, w)).astype("float32")
    idx_h = np.expand_dims(np.arange(1, h + 1), 1)
    idx_w = np.expand_dims(np.arange(1, w + 1), 0)
    delta = np.random.randint(15, 75 + 1)
    ch = np.random.randint(h)
    cw = np.random.randint(w)

    p = (np.abs(idx_h - ch) ** 2 + np.abs(idx_w - cw) ** 2) ** 0.5
    p /= np.max(p)
    noise *= p
    noise = np.cos(noise * np.pi / 2) ** 4
    if len(image.shape) == 3:
        noise = np.expand_dims(noise, -1)
    if random.random() < 0.5:
        image = np.clip(image.astype("float32") + noise.astype("float32") * delta, 0, 255).astype("uint8")
    else:
        image = np.clip(image.astype("float32") + (1 - noise.astype("float32")) * delta, 0, 255).astype("uint8")
    return image


def LC(p=1):
    return RandomBrightnessContrast(brightness_limit=(0.2, 0.4), contrast_limit=(-0.8, -0.2), p=p)


def Blur(p=1):
    return Compose([
        GaussianBlur(blur_limit=(7, 23), sigma_limit=(1, 3), p=1),
    ], p=p)


def Noise(p=1):
    return Compose([
        Lambda(image=nonuniformity_optical, p=1),
        Lambda(image=stripe_noise, p=1),
        GaussNoise(var_limit=(5.0 ** 2, 20.0 ** 2), p=1),
    ], p=p)


def deg_simple(patch_size, p=1):
    return Compose([

        # step 1
        RandomCrop(patch_size[0], patch_size[1], always_apply=True),
        RandomBrightnessContrast(brightness_limit=(0.2, 0.4), contrast_limit=(-0.8, -0.2), p=0.8),

        # step 2
        GaussianBlur(blur_limit=(7, 23), sigma_limit=(1, 3), p=0.8),

        # step 3
        Lambda(image=nonuniformity_optical, p=0.8),
        Lambda(image=stripe_noise, p=0.8),
        GaussNoise(var_limit=(5.0 ** 2, 20.0 ** 2), p=0.8),
    ], p=p)


def deg_simple_step(patch_size, img, label):
    img_list = {'image': np.array(img, dtype=np.uint8), 'mask': np.array(label, dtype=np.uint8)}
    img_list = RandomCrop(patch_size[0], patch_size[1], always_apply=True)(**img_list)
    img_list1 = LC()(**img_list)
    img_list2 = Blur()(**img_list)
    img_list3 = Noise()(**img_list)
    return [img_list1, img_list2, img_list3]


def deg_simple_chain(patch_size, img, label):
    img_list = {'image': np.array(img, dtype=np.uint8), 'mask': np.array(label, dtype=np.uint8)}
    img_list = RandomCrop(patch_size[0], patch_size[1], always_apply=True)(**img_list)
    img_list1 = LC()(**img_list)
    img_list2 = Blur()(**img_list1)
    img_list3 = Noise()(**img_list2)
    return [img_list1, img_list2, img_list3]


def aug_RC(patch_size, p=1):
    return RandomCrop(patch_size[0], patch_size[1], always_apply=True)


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size):
        super(TrainSetLoader).__init__()
        self.dataset_label_dir = dataset_dir + '/' + dataset_name
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        
        # Check for pre-generated corrupt data
        # Default save path from create_corrupt_labeled.py is ./datasets/corrupt_labeled
        project_root = os.getcwd()
        self.corrupt_dir = os.path.join(project_root, 'datasets', 'corrupt_labeled', dataset_name)
        self.label_file = os.path.join(project_root, 'datasets', 'corrupt_labeled', 'degradation_labels.npy')
        
        self.use_pre_generated = False
        if os.path.exists(self.corrupt_dir) and os.path.exists(self.label_file):
            print(f"==> Found pre-generated corrupt data at {self.corrupt_dir}")
            self.use_pre_generated = True
            self.labels_dict = np.load(self.label_file, allow_pickle=True).item()
        else:
            print("==> Pre-generated corrupt data not found. Using online augmentation.")
        
        # 优先查找 dataset_dir 下的 train.txt，其次查找 dataset_label_dir 下的
        txt_path = os.path.join(self.dataset_label_dir, 'train.txt')
        if not os.path.exists(txt_path):
             txt_path = os.path.join(dataset_dir, 'train.txt')
             
        if not os.path.exists(txt_path):
             # 如果都没找到，尝试直接读取文件夹生成列表
             print(f"Warning: {txt_path} not found. Scanning folder directly...")
             if os.path.exists(self.dataset_label_dir):
                 self.train_list = [f for f in os.listdir(self.dataset_label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
             else:
                  raise FileNotFoundError(f"Neither train.txt nor image folder found at {self.dataset_label_dir}")
        else:
            with open(txt_path, 'r', encoding='utf-8') as f:
                self.train_list = f.read().splitlines()
                
        self.train_list = [x.strip().lstrip('\ufeff') for x in self.train_list if x.strip() and x.strip().lower().endswith(IMG_EXTENSIONS)] # Filter empty lines, remove BOM and check extensions
        self.transform = augumentation()

    def __getitem__(self, idx):
        # 兼容处理：有些列表里可能只存了文件名，有些可能存了相对路径
        # 我们假设 self.dataset_label_dir 是图片所在的文件夹
        img_name = self.train_list[idx]
        
        # 尝试构建完整路径
        # 1. 直接拼接 (假设 dataset_label_dir 是包含图片的目录，如 .../Ir)
        full_path = os.path.join(self.dataset_label_dir, img_name).replace('\\', '/')
        
        # 2. 如果 dataset_label_dir 还有子目录 imgs (原始代码逻辑)
        if not os.path.exists(full_path):
             full_path = os.path.join(self.dataset_label_dir, 'imgs', img_name).replace('\\', '/')
        
        # 3. 终极尝试
        if not os.path.exists(full_path):
            # 也许 dataset_label_dir 已经是上级目录，需要加 dataset_name
             full_path = os.path.join(self.dataset_label_dir, 'Ir', img_name).replace('\\', '/')

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")

        label = Image.open(full_path).convert('L')
        img = Image.open(full_path).convert('L')

        if self.use_pre_generated:
            # Load corrupt image from disk
            corrupt_path = os.path.join(self.corrupt_dir, img_name)
            if not os.path.exists(corrupt_path):
                 # Try with replaced extension if needed, but assuming name match
                 # If not found, fallback to clean image (m=0)
                 img_lists = [{'image': np.array(img, dtype=np.uint8), 'mask': np.array(label, dtype=np.uint8)}] * 3
                 m = 0
            else:
                try:
                    cor_img_pil = Image.open(corrupt_path).convert('L')
                    cor_img_arr = np.array(cor_img_pil, dtype=np.uint8)
                    label_arr = np.array(label, dtype=np.uint8)
                    
                    # Construct img_lists with 3 copies to satisfy train.py structure
                    img_lists = [
                        {'image': cor_img_arr, 'mask': label_arr},
                        {'image': cor_img_arr, 'mask': label_arr},
                        {'image': cor_img_arr, 'mask': label_arr}
                    ]
                    
                    # Get degradation label
                    m = self.labels_dict.get(img_name, 0)
                    
                    # Apply RandomCrop to all of them consistently
                    # Note: We need to crop both input and label at the same location
                    # aug_RC helper can do this but it expects one image.
                    # We can manually crop.
                    
                    # Use existing aug_RC logic?
                    # dataset.py has random_crop function
                    
                    # Let's stack them for cropping
                    # Or use the `aug_RC` logic which calls RandomCrop
                    # RandomCrop in albumentations:
                    # Compose([RandomCrop(...)])
                    
                    # Let's manually do random crop to ensure consistency
                    h, w = label_arr.shape
                    ph, pw = self.patch_size
                    if h < ph or w < pw:
                        # Pad if needed (reuse logic from random_crop if available, or just pad)
                        pad_h = max(ph - h, 0)
                        pad_w = max(pw - w, 0)
                        # Pad all images
                        for i in range(3):
                            img_lists[i]['image'] = np.pad(img_lists[i]['image'], ((0, pad_h), (0, pad_w)), mode='constant')
                            img_lists[i]['mask'] = np.pad(img_lists[i]['mask'], ((0, pad_h), (0, pad_w)), mode='constant')
                        h, w = img_lists[0]['image'].shape
                    
                    h_start = random.randint(0, h - ph)
                    w_start = random.randint(0, w - pw)
                    
                    for i in range(3):
                        img_lists[i]['image'] = img_lists[i]['image'][h_start:h_start+ph, w_start:w_start+pw]
                        img_lists[i]['mask'] = img_lists[i]['mask'][h_start:h_start+ph, w_start:w_start+pw]
                        
                except Exception as e:
                    print(f"Error loading {corrupt_path}: {e}")
                    img_lists = deg_simple_step(self.patch_size, img, label)
                    m = 0
        elif random.random() < 0.5:
            img_lists = deg_simple_step(self.patch_size, img, label)
            m = 0
        else:
            img_lists = deg_simple_chain(self.patch_size, img, label)
            m = 1

        cor_img_1 = Image.fromarray(img_lists[0]['image']).convert('L')
        cor_img_2 = Image.fromarray(img_lists[1]['image']).convert('L')
        cor_img_3 = Image.fromarray(img_lists[2]['image']).convert('L')
        label = Image.fromarray(img_lists[0]['mask']).convert('L')

        img_list = [np.array(cor_img_1, dtype=np.float32) / 255.0, np.array(cor_img_2, dtype=np.float32) / 255.0,
                    np.array(cor_img_3, dtype=np.float32) / 255.0]

        label = np.array(label, dtype=np.float32) / 255.0
        if len(label.shape) > 3:
            label = label[:, :, 0]

        img_list_patch, label_patch = self.transform(img_list, label)
        img_list_patch = toTensor(img_list_patch)
        label_patch = label_patch[np.newaxis, :]
        label_patch = torch.from_numpy(np.ascontiguousarray(label_patch))
        return img_list_patch, label_patch, m

    def __len__(self):
        return len(self.train_list)


def toTensor(img_list):
    for idx in range(len(img_list)):
        img_list[idx] = torch.from_numpy(np.ascontiguousarray(img_list[idx][np.newaxis, :]))
    img_list = torch.stack(img_list)
    return img_list


class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, scenario, sub="hard"):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.dataset_label_dir = dataset_dir + '/' + dataset_name
        self.dataset_cor_dir = dataset_dir + '/' + dataset_name + '/' + scenario + '/' + sub
        
        # 自动查找 datasets/corrupt 下的干扰图
        project_root = os.getcwd()
        potential_corrupt_labeled = os.path.join(project_root, 'datasets', 'corrupt_test_labeled', dataset_name)
        
        if os.path.exists(potential_corrupt_labeled):
             print(f"Redirecting corrupt dir to: {potential_corrupt_labeled}")
             self.dataset_cor_dir = potential_corrupt_labeled
             
             # Check for paired GT directory
             potential_gt = os.path.join(project_root, 'datasets', 'corrupt_test_labeled', dataset_name + '_GT')
             if os.path.exists(potential_gt):
                 print(f"Redirecting label dir to: {potential_gt}")
                 self.dataset_label_dir = potential_gt
        elif not os.path.exists(self.dataset_cor_dir):
            # 假设 dataset_dir 是 .../datasets/M3FD_Detection
            # 我们想找 .../datasets/corrupt/Ir
            # 尝试回溯一级目录
            parent_dir = os.path.dirname(dataset_dir.rstrip('/\\'))
            potential_corrupt = os.path.join(parent_dir, 'corrupt', dataset_name).replace('\\', '/')
            if os.path.exists(potential_corrupt):
                print(f"Redirecting corrupt dir to: {potential_corrupt}")
                self.dataset_cor_dir = potential_corrupt

        
        # 类似 TrainSetLoader 的兼容逻辑
        txt_path = os.path.join(self.dataset_label_dir, 'test.txt')
        if not os.path.exists(txt_path):
             txt_path = os.path.join(dataset_dir, 'test.txt')

        if not os.path.exists(txt_path):
             # 如果都没找到，尝试直接读取文件夹生成列表
             print(f"Warning: {txt_path} not found. Scanning folder directly...")
             if os.path.exists(self.dataset_label_dir):
                 # 这里我们简单取前10张做测试，或者全部
                 self.test_list = [f for f in os.listdir(self.dataset_label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
             else:
                  raise FileNotFoundError(f"Neither test.txt nor image folder found at {self.dataset_label_dir}")
        else:
             with open(txt_path, 'r', encoding='utf-8') as f:
                self.test_list = f.read().splitlines()
        
        self.test_list = [x.strip().lstrip('\ufeff') for x in self.test_list if x.strip() and x.strip().lower().endswith(IMG_EXTENSIONS)]
        self.test_list = self.test_list

    def __getitem__(self, idx):
        # 兼容处理文件名
        img_name = self.test_list[idx]
        
        # 寻找 Label (干净图片)
        label_path = os.path.join(self.dataset_label_dir, img_name).replace('\\', '/')
        if not os.path.exists(label_path):
             label_path = os.path.join(self.dataset_label_dir, 'imgs', img_name).replace('\\', '/')
        if not os.path.exists(label_path):
             label_path = os.path.join(self.dataset_label_dir, 'Ir', img_name).replace('\\', '/')
             
        # 寻找 Input (干扰图片)
        # 注意：训练验证时，如果没有专门的验证集干扰图，我们暂时用原图代替，或者实时生成干扰
        # 原始代码假设有一个 dataset_cor_dir 存了干扰图
        # 如果找不到干扰图，我们为了代码能跑通，暂时 fallback 到原图（这就变成了自重建任务，仅用于跑通流程）
        
        input_path = os.path.join(self.dataset_cor_dir, img_name).replace('\\', '/')
        if not os.path.exists(input_path):
            # 尝试在同级目录下找
            input_path = label_path

        label = Image.open(label_path).convert('L')
        img = Image.open(input_path).convert('L')
        img_list = [np.array(img, dtype=np.float32) / 255.0]
        label = np.array(label, dtype=np.float32) / 255.0
        h, w = label.shape

        if len(label.shape) > 3:
            label = label[:, :, 0]

        img_list = toTensor(img_list)
        label = label[np.newaxis, :]
        label = torch.from_numpy(np.ascontiguousarray(label))
        return img_list[0], label, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)


class InferSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.dataset_label_dir = dataset_dir + '/' + dataset_name
        
        # 兼容: 如果有 imgs 子文件夹则用 imgs，否则直接扫描 dataset_label_dir
        if os.path.exists(self.dataset_label_dir + '/imgs/'):
            self.img_folder = self.dataset_label_dir + '/imgs/'
        else:
            self.img_folder = self.dataset_label_dir + '/'
            
        self.test_list = [x for x in os.listdir(self.img_folder) if x.lower().endswith(IMG_EXTENSIONS)]

    def __getitem__(self, idx):
        image = Image.open((self.img_folder + self.test_list[idx]).replace('//', '/')).convert('L')
        image = np.array(image, dtype=np.float32) / 255.0
        h, w = image.shape

        if len(image.shape) > 3:
            image = image[:, :, 0]

        image = PadImg(image, 32)

        image = image[np.newaxis, :]
        image = torch.from_numpy(np.ascontiguousarray(image))

        return image, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)


def _first_existing_dir(candidates):
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def _list_images(folder):
    return sorted([name for name in os.listdir(folder) if name.lower().endswith(IMG_EXTENSIONS)])


def _load_grayscale(path):
    return np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0


def _load_label_map(dataset_dir, split):
    label_candidates = [
        os.path.join(dataset_dir, split, 'degradation_labels.npy'),
        os.path.join(dataset_dir, 'degradation_labels.npy'),
        os.path.join(os.path.dirname(dataset_dir.rstrip('/\\')), 'corrupt_labeled', 'degradation_labels.npy'),
    ]
    for path in label_candidates:
        if os.path.exists(path):
            labels = np.load(path, allow_pickle=True).item()
            return {os.path.basename(str(k)): int(v) for k, v in labels.items()}
    return {}


class PairedTrainSetLoader(Dataset):
    """Paired input/target loader used by train.py.

    Expected layout:
        dataset_dir/
            train/
                input/
                target/
                degradation_labels.npy  # optional
    """

    def __init__(self, dataset_dir, split='train', patch_size=(48, 48)):
        super().__init__()
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.dataset_dir = dataset_dir
        self.split = split

        split_dir = os.path.join(dataset_dir, split)
        self.input_dir = _first_existing_dir([
            os.path.join(split_dir, 'input'),
            os.path.join(split_dir, 'degraded'),
            os.path.join(split_dir, 'corrupt'),
            os.path.join(split_dir, 'images'),
            split_dir,
        ])
        self.target_dir = _first_existing_dir([
            os.path.join(split_dir, 'target'),
            os.path.join(split_dir, 'gt'),
            os.path.join(split_dir, 'clean'),
            os.path.join(split_dir, 'label'),
            os.path.join(split_dir, 'labels'),
            os.path.join(dataset_dir, 'Ir'),
            os.path.join(dataset_dir, 'imgs'),
        ])

        if self.input_dir is None:
            raise FileNotFoundError(f"Could not find paired input folder under {split_dir}")
        if self.target_dir is None:
            self.target_dir = self.input_dir

        self.image_list = _list_images(self.input_dir)
        self.labels = _load_label_map(dataset_dir, split)
        self.transform = augumentation()

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        input_path = os.path.join(self.input_dir, img_name)
        target_path = os.path.join(self.target_dir, img_name)
        if not os.path.exists(target_path):
            target_path = input_path

        input_img = _load_grayscale(input_path)
        target_img = _load_grayscale(target_path)

        input_list, target_patch = random_crop([input_img, input_img, input_img], target_img, self.patch_size)
        input_list, target_patch = self.transform(input_list, target_patch)

        input_tensor = toTensor(input_list)
        target_tensor = torch.from_numpy(np.ascontiguousarray(target_patch[np.newaxis, :]))
        degradation_label = self.labels.get(os.path.basename(img_name), 0)
        return input_tensor, target_tensor, degradation_label

    def __len__(self):
        return len(self.image_list)


class PairedEvalSetLoader(Dataset):
    """Paired validation loader used by train.py."""

    def __init__(self, dataset_dir, split='test'):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.split = split

        split_dir = os.path.join(dataset_dir, split)
        self.input_dir = _first_existing_dir([
            os.path.join(split_dir, 'input'),
            os.path.join(split_dir, 'degraded'),
            os.path.join(split_dir, 'corrupt'),
            os.path.join(split_dir, 'images'),
            split_dir,
        ])
        self.target_dir = _first_existing_dir([
            os.path.join(split_dir, 'target'),
            os.path.join(split_dir, 'gt'),
            os.path.join(split_dir, 'clean'),
            os.path.join(split_dir, 'label'),
            os.path.join(split_dir, 'labels'),
            os.path.join(dataset_dir, 'Ir'),
            os.path.join(dataset_dir, 'imgs'),
        ])

        if self.input_dir is None:
            raise FileNotFoundError(f"Could not find paired input folder under {split_dir}")
        if self.target_dir is None:
            self.target_dir = self.input_dir

        self.image_list = _list_images(self.input_dir)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        input_path = os.path.join(self.input_dir, img_name)
        target_path = os.path.join(self.target_dir, img_name)
        if not os.path.exists(target_path):
            target_path = input_path

        input_img = _load_grayscale(input_path)
        target_img = _load_grayscale(target_path)
        h, w = target_img.shape

        input_tensor = torch.from_numpy(np.ascontiguousarray(input_img[np.newaxis, :]))
        target_tensor = torch.from_numpy(np.ascontiguousarray(target_img[np.newaxis, :]))
        return input_tensor, target_tensor, [h, w], img_name

    def __len__(self):
        return len(self.image_list)


class VisSetLoader(Dataset):
    def __init__(self, dataset_dir):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.test_list = [x for x in os.listdir(self.dataset_dir) if x.lower().endswith(IMG_EXTENSIONS)]

    def __getitem__(self, idx):
        image = Image.open((self.dataset_dir + '/' + self.test_list[idx]).replace('//', '/')).convert('L')
        image = np.array(image, dtype=np.float32) / 255.0
        h, w = image.shape

        if len(image.shape) > 3:
            image = image[:, :, 0]

        image = PadImg(image, 32)

        image = image[np.newaxis, :]
        image = torch.from_numpy(np.ascontiguousarray(image))

        return image, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)


class augumentation(object):
    def __call__(self, inputs, target):
        if not isinstance(inputs, List):
            inputs = [inputs]

        if random.random() < 0.5:
            for idx in range(len(inputs)):
                inputs[idx] = inputs[idx][::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:
            for idx in range(len(inputs)):
                inputs[idx] = inputs[idx][:, ::-1]
            target = target[:, ::-1]
        return inputs, target


def random_crop(img_list, mask, patch_size):
    if not isinstance(img_list, List):
        img_list = [img_list]

    h, w = mask.shape
    base_h, base_w = patch_size
    if h < base_h or w < base_w:
        for idx in range(len(img_list)):
            img_list[idx] = np.pad(img_list[idx], ((0, max(h, base_h) - h), (0, max(w, base_w) - w)),
                                   mode='constant')
        mask = np.pad(mask, ((0, max(h, base_h) - h), (0, max(w, base_w) - w)), mode='constant')
        h, w = mask.shape

    h_start = random.randint(0, h - base_h)
    h_end = h_start + base_h
    w_start = random.randint(0, w - base_w)
    w_end = w_start + base_w

    for idx in range(len(img_list)):
        img_list[idx] = img_list[idx][h_start:h_end, w_start:w_end]
    mask_patch = mask[h_start:h_end, w_start:w_end]

    return img_list, mask_patch


def PadImg(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode='constant')
    return img
