'''
Dataloader to process Adobe Image Matting Dataset.

From GCA_Matting(https://github.com/Yaoyi-Li/GCA-Matting/tree/master/dataloader)
'''
import os
import glob
import logging
import os.path as osp
import functools
import numpy as np
import torch
import cv2
import math
import numbers
import random
import pickle
from   torch.utils.data import Dataset, DataLoader
from   torch.nn import functional as F
from   torchvision import transforms
from easydict import EasyDict
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm

# Base default config
CONFIG = EasyDict({})

# Model config
CONFIG.model = EasyDict({})
# one-hot or class, choice: [3, 1]
CONFIG.model.trimap_channel = 1

# Dataloader config
CONFIG.data = EasyDict({})
# feed forward image size (untested)
CONFIG.data.crop_size = 512
# composition of two foregrounds, affine transform, crop and HSV jitter
CONFIG.data.cutmask_prob = 0.25
CONFIG.data.augmentation = True
CONFIG.data.random_interp = True

class Prefetcher():
    """
    Modified from the data_prefetcher in https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    """
    def __init__(self, loader):
        self.orig_loader = loader
        self.stream = torch.cuda.Stream()
        self.next_sample = None

    def preload(self):
        try:
            self.next_sample = next(self.loader)
        except StopIteration:
            self.next_sample = None
            return

        with torch.cuda.stream(self.stream):
            for key, value in self.next_sample.items():
                if isinstance(value, torch.Tensor):
                    self.next_sample[key] = value.cuda(non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        sample = self.next_sample
        if sample is not None:
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    sample[key].record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            # throw stop exception if there is no more data to perform as a default dataloader
            raise StopIteration("No samples in loader. example: `iterator = iter(Prefetcher(loader)); "
                                "data = next(iterator)`")
        return sample

    def __iter__(self):
        self.loader = iter(self.orig_loader)
        self.preload()
        return self


class ImageFile(object):
    def __init__(self, phase='train'):
        self.phase = phase
        self.rng = np.random.RandomState(0)

    def _get_valid_names(self, *dirs, shuffle=True):
        name_sets = [self._get_name_set(d) for d in dirs]

        def _join_and(a, b):
            return a & b

        valid_names = list(functools.reduce(_join_and, name_sets))
        if shuffle:
            self.rng.shuffle(valid_names)

        return valid_names

    @staticmethod
    def _get_name_set(dir_name):
        path_list = glob.glob(os.path.join(dir_name, '*'))
        name_set = set()
        for path in path_list:
            name = os.path.basename(path)
            name = os.path.splitext(name)[0]
            name_set.add(name)
        return name_set

    @staticmethod
    def _list_abspath(data_dir, ext, data_list):
        return [os.path.join(data_dir, name + ext)
                for name in data_list]

class ImageFileTrain(ImageFile):
    def __init__(self,
                 alpha_dir="train_alpha",
                 fg_dir="train_fg",
                 bg_dir="train_bg",
                 alpha_ext=".jpg",
                 fg_ext=".jpg",
                 bg_ext=".jpg",
                 root='',
                 ):
        super(ImageFileTrain, self).__init__(phase="train")

        self.alpha_dir  = alpha_dir
        self.fg_dir     = fg_dir
        self.bg_dir     = bg_dir
        self.alpha_ext  = alpha_ext
        self.fg_ext     = fg_ext
        self.bg_ext     = bg_ext
        logger = setup_logger(name=__name__)

        self.valid_fg_list = self._get_valid_names(self.fg_dir, self.alpha_dir)
        self.valid_bg_list = [os.path.splitext(name)[0] for name in os.listdir(self.bg_dir)]

        self.alpha = self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_fg_list)
        self.fg = self._list_abspath(self.fg_dir, self.fg_ext, self.valid_fg_list)
        self.bg = self._list_abspath(self.bg_dir, self.bg_ext, self.valid_bg_list)



    def __len__(self):
        return len(self.alpha)


class ImageFileTest(ImageFile):
    def __init__(self,
                 alpha_dir="test_alpha",
                 merged_dir="test_merged",
                 trimap_dir="test_trimap",
                 alpha_ext=".png",
                 merged_ext=".png",
                 trimap_ext=".png"):
        super(ImageFileTest, self).__init__(phase="test")

        self.alpha_dir  = alpha_dir
        self.merged_dir = merged_dir
        self.trimap_dir = trimap_dir
        self.alpha_ext  = alpha_ext
        self.merged_ext = merged_ext
        self.trimap_ext = trimap_ext

        self.valid_image_list = self._get_valid_names(self.alpha_dir, self.merged_dir, self.trimap_dir, shuffle=False)

        self.alpha = self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_image_list)
        self.merged = self._list_abspath(self.merged_dir, self.merged_ext, self.valid_image_list)
        self.trimap = self._list_abspath(self.trimap_dir, self.trimap_ext, self.valid_image_list)

    def __len__(self):
        return len(self.alpha)

interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]


def maybe_random_interp(cv2_interp):
    if CONFIG.data.random_interp:
        return np.random.choice(interp_list)
    else:
        return cv2_interp


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, phase="test"):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        self.phase = phase

    def __call__(self, sample):
        image, alpha, trimap, mask = sample['image'][:,:,::-1], sample['alpha'], sample['trimap'], sample['mask']
        
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
     
        image = image.transpose((2, 0, 1)).astype(np.float32)
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
        
        mask = np.expand_dims(mask.astype(np.float32), axis=0)

        image /= 255.

        if self.phase == "train":
            fg = sample['fg'][:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['fg'] = torch.from_numpy(fg)
            bg = sample['bg'][:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['bg'] = torch.from_numpy(bg)

        sample['image'], sample['alpha'], sample['trimap'] = \
            torch.from_numpy(image), torch.from_numpy(alpha), torch.from_numpy(trimap).to(torch.long)
        sample['image'] = sample['image']

        if CONFIG.model.trimap_channel == 3:
            sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2,0,1).float()
        elif CONFIG.model.trimap_channel == 1:
            sample['trimap'] = sample['trimap'][None,...].float()
        else:
            raise NotImplementedError("CONFIG.model.trimap_channel can only be 3 or 1")

        sample['mask'] = torch.from_numpy(mask).float()

        return sample


class RandomAffine(object):
    """
    Random affine translation
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        rows, cols, ch = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, fg.size)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, fg.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        fg = cv2.warpAffine(fg, M, (cols, rows),
                            flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        sample['fg'], sample['alpha'] = fg, alpha

        return sample


    @ staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix


class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):
        sample_ori = sample.copy()
        fg, alpha = sample['fg'], sample['alpha']
        # if alpha is all 0 skip
        if np.all(alpha==0):
            return sample_ori
        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        fg = cv2.cvtColor(fg.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = fg[:, :, 1][alpha > 0].mean()
        if np.isnan(sat_bar):
            return sample_ori
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat>1] = 2 - sat[sat>1]
        fg[:, :, 1] = sat
        # Value noise
        val_bar = fg[:, :, 2][alpha > 0].mean()
        if np.isnan(val_bar):
            return sample_ori
        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val>1] = 2 - val[val>1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
        sample['fg'] = fg*255

        return sample


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        if np.random.uniform(0, 1) < self.prob:
            fg = cv2.flip(fg, 1)
            alpha = cv2.flip(alpha, 1)
        sample['fg'], sample['alpha'] = fg, alpha

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=( CONFIG.data.crop_size, CONFIG.data.crop_size)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        fg, alpha, trimap, mask, name = sample['fg'],  sample['alpha'], sample['trimap'], sample['mask'], sample['image_name']
        bg = sample['bg']
        h, w = trimap.shape
        bg = cv2.resize(bg, (w, h), interpolation=maybe_random_interp(cv2.INTER_CUBIC))
        if w < self.output_size[0]+1 or h < self.output_size[1]+1:
            ratio = 1.1*self.output_size[0]/h if h < w else 1.1*self.output_size[1]/w
            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0]+1 or w < self.output_size[1]+1:
                fg = cv2.resize(fg, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha = cv2.resize(alpha, (int(w*ratio), int(h*ratio)),
                                   interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                trimap = cv2.resize(trimap, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST)
                bg = cv2.resize(bg, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_CUBIC))
                mask = cv2.resize(mask, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST)
                h, w = trimap.shape
        small_trimap = cv2.resize(trimap, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
        unknown_list = list(zip(*np.where(small_trimap[self.margin//4:(h-self.margin)//4,
                                                       self.margin//4:(w-self.margin)//4] == 128)))
        unknown_num = len(unknown_list)
        if len(unknown_list) < 10:
            left_top = (np.random.randint(0, h-self.output_size[0]+1), np.random.randint(0, w-self.output_size[1]+1))
        else:
            idx = np.random.randint(unknown_num)
            left_top = (unknown_list[idx][0]*4, unknown_list[idx][1]*4)

        fg_crop = fg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]
        alpha_crop = alpha[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        bg_crop = bg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]
        trimap_crop = trimap[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        mask_crop = mask[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]

        if len(np.where(trimap==128)[0]) == 0:
            self.logger.error("{} does not have enough unknown area for crop. Resized to target size."
                                "left_top: {}".format(name, left_top))
            fg_crop = cv2.resize(fg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(alpha, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            trimap_crop = cv2.resize(trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
            bg_crop = cv2.resize(bg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_CUBIC))
            mask_crop = cv2.resize(mask, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        sample.update({'fg': fg_crop, 'alpha': alpha_crop, 'trimap': trimap_crop, 'mask': mask_crop, 'bg': bg_crop})
        return sample


class OriginScale(object):
    def __call__(self, sample):
        h, w = sample["alpha_shape"]

        if h % 32 == 0 and w % 32 == 0:
            return sample

        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w

        padded_image = np.pad(sample['image'], ((0,pad_h), (0, pad_w), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((0,pad_h), (0, pad_w)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((0,pad_h), (0, pad_w)), mode="reflect")

        sample['image'] = padded_image
        sample['trimap'] = padded_trimap
        sample['mask'] = padded_mask

        return sample


class GenMask(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]

    def __call__(self, sample):
        alpha_ori = sample['alpha']
        h, w = alpha_ori.shape

        max_kernel_size = 30
        alpha = cv2.resize(alpha_ori, (640,640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        ### generate trimap
        fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])

        fg_width = np.random.randint(1, 30)
        bg_width = np.random.randint(1, 30)
        fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        trimap = cv2.resize(trimap, (w,h), interpolation=cv2.INTER_NEAREST)
        sample['trimap'] = trimap

        ### generate mask
        low = 0.01
        high = 1.0
        thres = random.random() * (high - low) + low
        seg_mask = (alpha >= thres).astype(np.int).astype(np.uint8)
        random_num = random.randint(0,3)
        if random_num == 0:
            seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        elif random_num == 1:
            seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        elif random_num == 2:
            seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
            seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        elif random_num == 3:
            seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
            seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        
        seg_mask = cv2.resize(seg_mask, (w,h), interpolation=cv2.INTER_NEAREST)
        sample['mask'] = seg_mask

        return sample


class Composite(object):
    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
        fg[fg < 0 ] = 0
        fg[fg > 255] = 255
        bg[bg < 0 ] = 0
        bg[bg > 255] = 255

        image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
        sample['image'] = image
        return sample


class CutMask(object):
    def __init__(self, perturb_prob = 0):
        self.perturb_prob = perturb_prob

    def __call__(self, sample):
        if np.random.rand() < self.perturb_prob:
            return sample

        mask = sample['mask'] # H x W, trimap 0--255, segmask 0--1, alpha 0--1
        h, w = mask.shape
        perturb_size_h, perturb_size_w = random.randint(h // 4, h // 2), random.randint(w // 4, w // 2)
        x = random.randint(0, h - perturb_size_h)
        y = random.randint(0, w - perturb_size_w)
        x1 = random.randint(0, h - perturb_size_h)
        y1 = random.randint(0, w - perturb_size_w)
        
        mask[x:x+perturb_size_h, y:y+perturb_size_w] = mask[x1:x1+perturb_size_h, y1:y1+perturb_size_w].copy()
        
        sample['mask'] = mask
        return sample


class DataGenerator(Dataset):
    def __init__(self, data, phase="train"):
        self.phase = phase
        self.crop_size = CONFIG.data.crop_size
        self.alpha = data.alpha

        if self.phase == "train":
            self.fg = data.fg
            self.bg = data.bg
            self.merged = []
            self.trimap = []

        else:
            self.fg = []
            self.bg = []
            self.merged = data.merged
            self.trimap = data.trimap

        train_trans = [
            RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
            GenMask(),
            CutMask(perturb_prob=CONFIG.data.cutmask_prob),
            RandomCrop((self.crop_size, self.crop_size)),
            RandomJitter(),
            Composite(),
            ToTensor(phase="train") ]

        test_trans = [ OriginScale(), ToTensor() ]

        self.transform = {
            'train':
                transforms.Compose(train_trans),
            'val':
                transforms.Compose([
                    OriginScale(),
                    ToTensor()
                ]),
            'test':
                transforms.Compose(test_trans)
        }[phase]

        self.fg_num = len(self.fg)

    def __getitem__(self, idx):
        if self.phase == "train":
            fg = cv2.imread(self.fg[idx % self.fg_num])
            alpha = cv2.imread(self.alpha[idx % self.fg_num], 0).astype(np.float32)/255
            bg = cv2.imread(self.bg[idx], 1)

            fg, alpha = self._composite_fg(fg, alpha, idx)

            image_name = os.path.split(self.fg[idx % self.fg_num])[-1]
            sample = {'fg': fg, 'alpha': alpha, 'bg': bg, 'image_name': image_name}

        else:
            image = cv2.imread(self.merged[idx])
            alpha = cv2.imread(self.alpha[idx], 0)/255.
            trimap = cv2.imread(self.trimap[idx], 0)
            mask = (trimap >= 170).astype(np.float32)
            image_name = os.path.split(self.merged[idx])[-1]

            sample = {'image': image, 'alpha': alpha, 'trimap': trimap, 'mask': mask, 'image_name': image_name, 'alpha_shape': alpha.shape}

        sample = self.transform(sample)

        return sample

    def _composite_fg(self, fg, alpha, idx):

        if np.random.rand() < 0.5:
            idx2 = np.random.randint(self.fg_num) + idx
            fg2 = cv2.imread(self.fg[idx2 % self.fg_num])
            alpha2 = cv2.imread(self.alpha[idx2 % self.fg_num], 0).astype(np.float32)/255.
            h, w = alpha.shape
            fg2 = cv2.resize(fg2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha2 = cv2.resize(alpha2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

            alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
            if  np.any(alpha_tmp < 1):
                fg = fg.astype(np.float32) * alpha[:,:,None] + fg2.astype(np.float32) * (1 - alpha[:,:,None])
                # The overlap of two 50% transparency should be 25%
                alpha = alpha_tmp
                fg = fg.astype(np.uint8)

        if np.random.rand() < 0.25:
            fg = cv2.resize(fg, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha = cv2.resize(alpha, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        return fg, alpha

    def __len__(self):
        if self.phase == "train":
            return len(self.bg)
        else:
            return len(self.alpha)