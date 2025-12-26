from tqdm import tqdm
import copy
import os
import json
import pickle
import glob
import random
import logging
from functools import partial
from typing import Sequence, Dict, Union, List
from dataclasses import dataclass
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset
import transformers
from datasets import load_dataset
from torchvision import transforms
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop

logger = logging.getLogger(__name__)
# Also use the main logger to ensure visibility in training logs
_main_logger = logging.getLogger("__main__")

OPENAI_DATASET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
OPENAI_DATASET_STD = np.array([0.26862954, 0.26130258, 0.27577711])
DEFAULT_IMAGE_FILE_SUFFIX = ['jpg', '0.jpg', '0.png', 'png', 'jpeg', '0.jpeg', 'webp']

def find_image(sample):
    for suffix in DEFAULT_IMAGE_FILE_SUFFIX:
        if suffix in sample.keys():
            sample['0.jpg'] = sample[suffix]
            break
    return sample

def select_dataset_files_by_ratio(dataset_path: str, dataset_ratio: float = 1.0, seed: int = 42) -> Union[str, List[str]]:
    """
    根据指定比例从数据集文件夹中选择tar文件。
    
    Args:
        dataset_path: 数据集路径（目录或包含通配符的路径）
        dataset_ratio: 数据集比例，支持 0.25, 0.5, 0.75, 1.0
        seed: 随机种子，用于保证每次选择相同的文件
        
    Returns:
        如果ratio=1.0，返回原始路径字符串；否则返回选中的tar文件列表
    """
    if dataset_ratio is None or dataset_ratio >= 1.0:
        return dataset_path
    
    # 处理通配符路径
    if '*' in dataset_path:
        # 使用glob查找所有匹配的tar文件
        tar_files = sorted(glob.glob(dataset_path))
        if not tar_files:
            # 如果通配符匹配失败，尝试在目录中查找所有tar文件
            base_dir = os.path.dirname(dataset_path.split('*')[0])
            if os.path.exists(base_dir):
                tar_files = sorted(glob.glob(os.path.join(base_dir, "*.tar")))
    else:
        # 如果是目录路径，查找目录中所有tar文件
        if os.path.isdir(dataset_path):
            tar_files = sorted(glob.glob(os.path.join(dataset_path, "*.tar")))
            # 如果目录中没有tar文件，尝试递归查找
            if not tar_files:
                tar_files = sorted(glob.glob(os.path.join(dataset_path, "**", "*.tar"), recursive=True))
        else:
            # 如果是单个文件路径
            if dataset_path.endswith('.tar') and os.path.exists(dataset_path):
                tar_files = [dataset_path]
            else:
                logger.warning(f"Could not find tar files, using original path: {dataset_path}")
                return dataset_path
    
    if not tar_files:
        logger.warning(f"No tar files found in {dataset_path}, using original path")
        return dataset_path
    
    # 使用固定种子随机选择指定比例的文件
    random.seed(seed)
    num_files_to_select = max(1, int(len(tar_files) * dataset_ratio))
    selected_files = random.sample(tar_files, num_files_to_select)
    selected_files = sorted(selected_files)  # 排序以保持一致性
    
    # Use logger to output information to training logs
    msg = f"Dataset file selection: selected {len(selected_files)} out of {len(tar_files)} files ({dataset_ratio*100:.1f}%)"
    logger.info(msg)
    _main_logger.info(msg)
    msg = f"Dataset path: {dataset_path}"
    logger.info(msg)
    _main_logger.info(msg)
    msg = f"Selected files: {len(selected_files)} / {len(tar_files)}"
    logger.info(msg)
    _main_logger.info(msg)
    if len(selected_files) <= 10:
        # If file count is small, list all selected files
        for i, fname in enumerate(selected_files, 1):
            msg = f"  File {i}: {os.path.basename(fname)}"
            logger.info(msg)
            _main_logger.info(msg)
    else:
        # If file count is large, only show first 5 and last 5
        for i, fname in enumerate(selected_files[:5], 1):
            msg = f"  File {i}: {os.path.basename(fname)}"
            logger.info(msg)
            _main_logger.info(msg)
        msg = f"  ... (and {len(selected_files) - 10} more files) ..."
        logger.info(msg)
        _main_logger.info(msg)
        for i, fname in enumerate(selected_files[-5:], len(selected_files) - 4):
            msg = f"  File {i}: {os.path.basename(fname)}"
            logger.info(msg)
            _main_logger.info(msg)
    
    return selected_files

# remove_columns=['__key__', '__url__', '0.txt', 'original_prompt']
def get_wds_dataset_and_collator(data_args, model_args):
    img_size = model_args.image_size
    train_processor = image_transform(img_size, is_train=True)
    val_processor = image_transform(img_size, is_train=False)
    
    # 根据dataset_ratio选择tar文件
    dataset_ratio = getattr(data_args, 'dataset_ratio', 1.0)
    msg = f"Loading dataset from: {data_args.dataset_path}, dataset ratio: {dataset_ratio}"
    logger.info(msg)
    _main_logger.info(msg)
    selected_files = select_dataset_files_by_ratio(data_args.dataset_path, dataset_ratio, seed=getattr(data_args, 'seed', 42))
    
    if isinstance(selected_files, list):
        msg = f"Loading dataset with {len(selected_files)} tar files"
        logger.info(msg)
        _main_logger.info(msg)
        data = load_dataset("webdataset", data_files=selected_files, split="train", streaming=True)
    else:
        msg = f"Loading dataset using full dataset path: {selected_files}"
        logger.info(msg)
        _main_logger.info(msg)
        data = load_dataset("webdataset", data_dir=selected_files, split="train", streaming=True)
    data = data.shuffle(buffer_size=2_000, seed=getattr(data_args, 'seed', 42))
    msg = "Dataset loading completed"
    logger.info(msg)
    _main_logger.info(msg)

    def decode(sample, img_processor):
        sample = find_image(sample)
        sample['0.jpg'] = img_processor(sample['0.jpg'])
        return sample
    data = data.map(
        partial(decode, img_processor=train_processor),
        # remove_columns=['__key__', '__url__', '0.txt', 'seg.txt']
        remove_columns=['__key__', '__url__']
    )
    data = data.filter(lambda sample: '0.jpg' in sample) # filter return samples that match the given condition
    data = data.rename_columns({'0.jpg': 'image'})
    data_collator = WebdatasetCollator(model_args.patch_size)
    
    return data, data_collator

def image_transform_original_resolution(
    image,
    patch_size: int,
):
    """accept a pil image and transform into torch.tensor"""
    w, h = map(lambda x: x // patch_size * patch_size, image.size)
    if w > 1024:
        h = int(h / (w / 1024) // patch_size * patch_size)
        w = 1024
    elif h > 1024:
        w = int(w / (h / 1024) // patch_size * patch_size)
        h = 1024
    def _convert_to_rgb(image):
        return image.convert('RGB')
    normalize = transforms.Normalize(
        mean=OPENAI_DATASET_MEAN, 
        std=OPENAI_DATASET_STD
    )
    transform = transforms.Compose([
        transforms.CenterCrop((h, w)),
        _convert_to_rgb,
        transforms.ToTensor(),
        normalize,
    ])
    ph, pw = h // patch_size, w // patch_size
    return transform(image), (ph, pw)

def get_wds_dataset_and_collator_arbitrary_resolution(data_args, model_args):
    # 根据dataset_ratio选择tar文件
    dataset_ratio = getattr(data_args, 'dataset_ratio', 1.0)
    msg = f"Loading dataset from: {data_args.dataset_path}, dataset ratio: {dataset_ratio}"
    logger.info(msg)
    _main_logger.info(msg)
    selected_files = select_dataset_files_by_ratio(data_args.dataset_path, dataset_ratio, seed=getattr(data_args, 'seed', 42))
    
    if isinstance(selected_files, list):
        msg = f"Loading dataset with {len(selected_files)} tar files"
        logger.info(msg)
        _main_logger.info(msg)
        data = load_dataset("webdataset", data_files=selected_files, split="train", streaming=True)
    else:
        msg = f"Loading dataset using full dataset path: {selected_files}"
        logger.info(msg)
        _main_logger.info(msg)
        data = load_dataset("webdataset", data_dir=selected_files, split="train", streaming=True)
    data = data.shuffle(buffer_size=2_000, seed=getattr(data_args, 'seed', 42))
    msg = "Dataset loading completed"
    logger.info(msg)
    _main_logger.info(msg)

    def decode_sample(sample, img_processor):
        sample = find_image(sample)
        sample['0.jpg'], sample['size'] = img_processor(sample['0.jpg'])
        return sample
    
    data = data.map(
        partial(
            decode_sample, 
            img_processor=partial(image_transform_original_resolution, patch_size=model_args.patch_size)
        ),
        remove_columns=['__key__', '__url__']
    )
    data = data.filter(lambda sample: '0.jpg' in sample and sample['0.jpg'].ndim == 3 and sample['0.jpg'].shape[-1] > 0 and sample['0.jpg'].shape[-2] > 0) # filter return samples that match the given condition
    data = data.rename_columns({'0.jpg': 'image'})
    data_collator = WebdatasetCollator(model_args.patch_size)
    
    return data, data_collator

def dataset_test(data_args, model_args):
    from datasets import load_dataset
    import numpy as np
    from functools import partial
    from torchvision import transforms
    OPENAI_DATASET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
    OPENAI_DATASET_STD = np.array([0.26862954, 0.26130258, 0.27577711])
    DEFAULT_IMAGE_FILE_SUFFIX = ['jpg', '0.jpg', '0.png', 'png', 'jpeg', '0.jpeg', 'webp']
    data = load_dataset("webdataset", data_dir="/share/project/datasets/laion-high-resolution/*/*.tar", split="train", streaming=True)
    data = data.shuffle(buffer_size=2_000, seed=1)
    
    data_iter = iter(data)

    def decode_sample(sample, img_processor):
        sample = find_image(sample)
        sample['0.jpg'], sample['size'] = img_processor(sample['0.jpg'])
        return sample

    def image_transform_original_resolution(
        image,
        patch_size: int,
    ):
        w, h = map(lambda x: x // patch_size * patch_size, image.size)
        def _convert_to_rgb(image):
            return image.convert('RGB')
        normalize = transforms.Normalize(
            mean=OPENAI_DATASET_MEAN, 
            std=OPENAI_DATASET_STD
        )
        transform = transforms.Compose([
            transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
            _convert_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])
        return transform(image)
    
    data = data.map(
        partial(
            decode_sample, 
            img_processor=partial(image_transform_original_resolution, patch_size=16)
        ),
        remove_columns=['__key__', '__url__']
    )
    data = data.filter(lambda sample: '0.jpg' in sample) # filter return samples that match the given condition
    data = data.rename_columns({'0.jpg': 'image'})
    data_collator = WebdatasetCollator()
    
    return data, data_collator

def collate_anyres(images, sizes, patch_size):
    """
    Args:
    * images: list of images
    * sizes: list of image sizes in (ph, pw), i.e., number of patches in h and w
    
    Return: args accepted by VQModel
    * pixel_values: packed images
    * cu_seqlens_img: 
    * max_seqlen_img
    * grid_hw
    * image_sizes
    """
    b, c = len(images), images[0].shape[0]
    max_patch_num = 1024 // patch_size

    image_sizes = torch.tensor([(image.shape[1], image.shape[2]) for image in images])
    H, W = image_sizes.max(dim=0).values
    padded_images = images[0].new_zeros(size=(b, c, H.item(), W.item()))

    h, w = torch.tensor(sizes).max(dim=0).values
    padding_masks = torch.zeros(size=(b, h.item(), w.item()), dtype=torch.bool)

    for i, (image, mask_size) in enumerate(zip(images, sizes)):
        padded_images[i, :, : image.shape[1], : image.shape[2]].copy_(image)
        padding_masks[i, : mask_size[0], : mask_size[1]] = 1

    padded_images = padded_images.reshape(b, c, h, patch_size, w, patch_size)
    padded_images = torch.einsum("nchpwq->nhwpqc", padded_images)
    padded_images = padded_images.reshape(b, h, w, -1)
    packed_images = padded_images[padding_masks]

    seq_lens = padding_masks.flatten(1, 2).sum(dim=-1)
    cu_seqlens_img = torch.nn.functional.pad(
        torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
    )
    max_seqlen_img = seq_lens.max()

    grid_h = torch.arange(0, h)[None, :, None].repeat(b, 1, w)
    grid_w = torch.arange(0, w)[None, None, :].repeat(b, h, 1)
    grid_hw = grid_h[padding_masks] * max_patch_num + grid_w[padding_masks]
    
    return packed_images, cu_seqlens_img, max_seqlen_img, grid_hw, torch.tensor(sizes)

@dataclass
class WebdatasetCollator:
    patch_size: int
    def __call__(self, samples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        images = [sample["image"] for sample in samples]
        if "size" in samples[0]:
            sizes = [sample['size'] for sample in samples]
        
        batch = {}

        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['pixel_values'] = torch.stack(images)
        else:
            batch['pixel_values'], batch['cu_seqlens_img'], \
                batch['max_seqlen_img'], batch['grid_hw'], \
                    batch['image_sizes'] = collate_anyres(images, sizes, self.patch_size)
        
        # print(f"{[image.shape for image in batch['pixel_values']]=}")
        return batch

def anyres_process_images_for_model(image_path=None, pil_image=None, patch_size=32):
    """
    given a list of image_path or pil_image, transform to input to model
    """
    if image_path is not None:
        assert pil_image is None
        if not isinstance(image_path, list):
            image_path = [image_path]
        pil_image = []
        for p in image_path:
            pil_image.append(Image.open(p).convert('RGB'))
    if not isinstance(pil_image, list):
        pil_image = [pil_image]
    
    if len(pil_image) % 2 != 0:
        pil_image.append(pil_image[-1])
    
    image_tensors, sizes = [], []
    for pil_i in pil_image:
        image_tensor, size = image_transform_original_resolution(image=pil_i, patch_size=patch_size)
        image_tensors.append(image_tensor)
        sizes.append(size)
    
    pixel_values, cu_seqlens_img, max_seqlen_img, grid_hw, image_sizes = collate_anyres(image_tensors, sizes, patch_size)
    
    return {
        'pixel_values': pixel_values,
        'cu_seqlens_img': cu_seqlens_img,
        'max_seqlen_img': max_seqlen_img,
        'grid_hw': grid_hw, 
        'image_sizes': image_sizes
    }

def get_in1k_dataset(data_args, model_args):
    import torchvision
    transform = image_transform(model_args.image_size, is_train=False)
    dataset = torchvision.datasets.ImageFolder(root="/share/project/datasets/ImageNet/val", transform=transform)
    def in1k_collator(samples):
        if model_args.gan_loss_weight:
            return {"pixel_values": torch.stack([sample[0] for sample in samples]), "optimizer_idx": 0}
        return {"pixel_values": torch.stack([sample[0] for sample in samples])}
    def in1k_collator_anyres(samples):
        images = [sample[0] for sample in samples]
        sizes = [[image.shape[1] // model_args.patch_size, image.shape[2] // model_args.patch_size] for image in images]
        b, c = len(images), images[0].shape[0]
        max_patch_num = 1024 // model_args.patch_size
        
        image_sizes = torch.tensor([(image.shape[1], image.shape[2]) for image in images])
        H, W = image_sizes.max(dim=0).values
        padded_images = images[0].new_zeros(size=(b, c, H.item(), W.item()))

        h, w = torch.tensor(sizes).max(dim=0).values
        padding_masks = torch.zeros(size=(b, h.item(), w.item()), dtype=torch.bool)
        
        for i, (image, mask_size) in enumerate(zip(images, sizes)):
            padded_images[i, :, : image.shape[1], : image.shape[2]].copy_(image)
            padding_masks[i, : mask_size[0], : mask_size[1]] = 1
        
        padded_images = padded_images.reshape(b, c, h, model_args.patch_size, w, model_args.patch_size)
        padded_images = torch.einsum("nchpwq->nhwpqc", padded_images)
        padded_images = padded_images.reshape(b, h, w, -1)
        packed_images = padded_images[padding_masks]
        
        seq_lens = padding_masks.flatten(1, 2).sum(dim=-1)
        cu_seqlens_img = torch.nn.functional.pad(
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
        )
        max_seqlen_img = seq_lens.max()
        
        grid_h = torch.arange(0, h)[None, :, None].repeat(b, 1, w)
        grid_w = torch.arange(0, w)[None, None, :].repeat(b, h, 1)
        grid_hw = grid_h[padding_masks] * max_patch_num + grid_w[padding_masks]
        
        batch = {}
        batch['pixel_values'] = packed_images
        batch['cu_seqlens_img'] = cu_seqlens_img
        batch['max_seqlen_img'] = max_seqlen_img
        batch['grid_hw'] = grid_hw
        batch['image_sizes'] = torch.tensor(sizes)
        if model_args.gan_loss_weight:
            batch["optimizer_idx"] = 0
        return batch
    # dataset = load_dataset("imagefolder", data_dir="/share/project/qiying/datasets/ImageNet/ImageNet/val")['validation']
    # transform = image_transform(model_args.image_size, is_train=False)
    # def transforms(examples):
    #     examples["pixel_values"] = [transform(image) for image in examples["pixel_values"]]
    #     return examples
    # dataset.set_transform(transforms)
    # dataset = dataset.remove_columns(["label"])
    # dataset = dataset.rename_column('image', 'pixel_values')
    return dataset, in1k_collator_anyres if data_args.arbitrary_resolution else in1k_collator
    
def get_highres_eval_dataset(data_args, model_args):
    # data = load_dataset("webdataset", data_dir="/share/project/datasets/laion-high-resolution/eval/eval_*.tar", split="train", streaming=True)

    # def decode_sample(sample, img_processor):
    #     sample = find_image(sample)
    #     sample['0.jpg'], sample['size'] = img_processor(sample['0.jpg'])
    #     return sample
    
    # data = data.map(
    #     partial(
    #         decode_sample, 
    #         img_processor=partial(image_transform_original_resolution, patch_size=model_args.patch_size)
    #     ),
    #     remove_columns=['__key__', '__url__'],
    # )
    # data = data.filter(lambda sample: '0.jpg' in sample and sample['0.jpg'].ndim == 3 and sample['0.jpg'].shape[-1] > 0 and sample['0.jpg'].shape[-2] > 0) # filter return samples that match the given condition
    # data = data.rename_columns({'0.jpg': 'image'})
    data = HighresEvalDataset()
    data_collator = WebdatasetCollator(model_args.patch_size)
    
    return data, data_collator

def image_transform(
    image_size: int,
    is_train: bool,
):
    mean = OPENAI_DATASET_MEAN
    std = OPENAI_DATASET_STD
    def _convert_to_rgb(image):
        return image.convert('RGB')
    normalize = transforms.Normalize(mean=mean, std=std)
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            _convert_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            _convert_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])


def norm_vq_img(img):
    arr = np.array(img)
    arr = arr.astype(np.float32) / 127.5 - 1
    img = torch.from_numpy(np.transpose(arr, [2, 0, 1]))
    return img


def prepare_image(img):
    """ Transform and normalize PIL Image to tensor. """
    transform = transforms.Compose([
            transforms.RandomResizedCrop(512, scale=(1., 1.), ratio=(1., 1.), interpolation=InterpolationMode.BICUBIC),
        ])
    pil_image = transform(img)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    return torch.from_numpy(np.transpose(arr, [2, 0, 1]))


def image_transform_for_vq(
    image_size: int,
    is_train: bool,
):

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    if is_train:
        return Compose([
            RandomResizedCrop(image_size, scale=(1., 1.), ratio=(1., 1.), interpolation=InterpolationMode.BICUBIC),
            _convert_to_rgb,
            norm_vq_img,
        ])
    else:
        transforms = [
            RandomResizedCrop(image_size, scale=(1., 1.), ratio=(1., 1.), interpolation=InterpolationMode.BICUBIC),
            _convert_to_rgb,
            norm_vq_img
        ]
        return Compose(transforms)

def split_val_set():
    data = load_dataset("webdataset", data_dir="/share/project/datasets/laion-high-resolution/*/*.tar", split="train", streaming=True)
    data = data.shuffle(buffer_size=100_000, seed=100)
    val_set = []
    val_ids = set()
    for i, item in enumerate(data):
        item_id = item['__url__'] + item['__key__']
        if i == 50_000:
            break
        val_set.append(item)
        val_ids.add(item_id)
    # with open("/share/project/datasets/laion-high-resolution/50k_eval.pkl", "wb") as f:
    #     pickle.dump(val_set, f)
    with open("/share/project/datasets/laion-high-resolution/50k_eval_ids.pkl", "wb") as f:
        pickle.dump(val_ids, f)
        
    import webdataset as wds
    from PIL import Image
    from pathlib import Path
    for i in range(50):
        sink = wds.TarWriter(f"/share/project/datasets/laion-high-resolution/eval/eval_{i}.tar")
        for sample in val_set[i * 1000: (i + 1) * 1000]:
            sink.write(sample)
        sink.close()

def preprocess_val_data():
    data = load_dataset("webdataset", data_dir="/share/project/datasets/laion-high-resolution/eval/*.tar", split="train", streaming=True)
    save_dir = "/share/project/datasets/laion-high-resolution/eval/"
    info = []
    def decode_sample(sample):
        sample = find_image(sample)
        sample['0.jpg'], sample['size'] = image_transform_original_resolution(sample['0.jpg'], patch_size=32)
        return sample
    for i, item in tqdm(enumerate(data)):
        image_file = save_dir + f"image_{i}.pkl"
        # process image -> size, image pkl
        item = decode_sample(item)
        with open(image_file, "wb") as f:
            pickle.dump(item['0.jpg'], f)
        info.append({
            "image_path": image_file,
            "size": item['size']
        })
        print(i)
    with open(save_dir + f"image_info.pkl", "wb") as f:
        pickle.dump(info, f)
        
class HighresEvalDataset(Dataset):
    def __init__(self):
        with open("/share/project/datasets/laion-high-resolution/eval/image_info.pkl", "rb") as f:
            self.info = pickle.load(f)
    
    def __getitem__(self, index):
        info = self.info[index]
        image_path, size = info['image_path'], info['size']
        with open(image_path, "rb") as f:
            image = pickle.load(f)
        return {"image": image, "size": size}
    
    def __len__(self):
        return len(self.info)

# preprocess_val_data()
# split_val_set()

    # def find_image(sample):
    #     for suffix in DEFAULT_IMAGE_FILE_SUFFIX:
    #         if suffix in sample.keys():
    #             sample['0.jpg'] = sample[suffix]
    #             break
    #     return sample

    # def decode_sample(sample, img_processor):
    #     sample = find_image(sample)
    #     sample['0.jpg'], sample['size'] = img_processor(sample['0.jpg'])
    #     return sample
    
    # data = data.map(
    #     partial(
    #         decode_sample, 
    #         img_processor=partial(image_transform_original_resolution, patch_size=model_args.patch_size)
    #     ),
    #     remove_columns=['__key__', '__url__']
    # )
    # data = data.filter(lambda sample: '0.jpg' in sample and sample['0.jpg'].ndim == 3 and sample['0.jpg'].shape[-1] > 0 and sample['0.jpg'].shape[-2] > 0) # filter return samples that match the given condition
    # data = data.rename_columns({'0.jpg': 'image'})
    # data_collator = WebdatasetCollator(model_args.patch_size)
    
    # return data, data_collator