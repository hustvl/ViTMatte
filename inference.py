'''
Inference for Composition-1k Dataset.

Run:
python inference.py \
    --config-dir path/to/config
    --checkpoint-dir path/to/checkpoint
    --inference-dir path/to/inference
    --data-dir path/to/data
'''
import os
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from os.path import join as opj
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser

import warnings
warnings.filterwarnings('ignore')

#Dataset and Dataloader
def collate_fn(batched_inputs):
    rets = dict()
    for k in batched_inputs[0].keys():
        rets[k] = torch.stack([_[k] for _ in batched_inputs])
    return rets

class Composition_1k(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_names = sorted(os.listdir(opj(self.data_dir, 'merged')))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        phas = Image.open(opj(self.data_dir, 'alpha_copy', self.file_names[idx]))
        tris = Image.open(opj(self.data_dir, 'trimaps', self.file_names[idx]))
        imgs = Image.open(opj(self.data_dir, 'merged', self.file_names[idx]))
        sample = {}

        sample['trimap'] = F.to_tensor(tris)
        sample['image'] = F.to_tensor(imgs)
        sample['image_name'] = self.file_names[idx]

        return sample

#model and output
def matting_inference(
    config_dir='',
    checkpoint_dir='',
    inference_dir='',
    data_dir='',
):
    #initializing model
    cfg = LazyConfig.load(config_dir)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model.eval()
    DetectionCheckpointer(model).load(checkpoint_dir)

    #initializing dataset
    composition_1k_dataloader = DataLoader(
    dataset = Composition_1k(
        data_dir = data_dir
    ),
    shuffle = False,
    batch_size = 1,
    # collate_fn = collate_fn,
    )
    
    #inferencing
    os.makedirs(inference_dir, exist_ok=True)

    for data in tqdm(composition_1k_dataloader):
        with torch.no_grad():
            for k in data.keys():
                if k == 'image_name':
                    continue
                else:
                    data[k].to(model.device)
            output = model(data)['phas'].flatten(0, 2)
            output = F.to_pil_image(output)
            output.save(opj(inference_dir, data['image_name'][0]))
            torch.cuda.empty_cache()

if __name__ == '__main__':
    #add argument we need:
    parser = default_argument_parser()
    parser.add_argument('--config-dir', type=str, required=True)
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--inference-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    
    args = parser.parse_args()
    matting_inference(
        config_dir = args.config_dir,
        checkpoint_dir = args.checkpoint_dir,
        inference_dir = args.inference_dir,
        data_dir = args.data_dir
    )