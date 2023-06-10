"""
It is used to run the model on one image (and its corresponding trimap).

For default, run:
python run_one_image.py \
    --config-dir configs/ViTMatte_S_100ep.py \
    --checkpoint-dir path/to/checkpoint
It will be saved in the directory ``./demo``.

If you want to run on your own image, run:
python run_one_image.py \
    --config-dir <your config directory> \
    --checkpoint-dir <your checkpoint directory> \
    --image-dir <your image directory> \
    --trimap-dir <your trimap directory> \
    --output-dir <your output directory>
"""
import os
from PIL import Image
from os.path import join as opj
from torchvision.transforms import functional as F
from detectron2.engine import default_argument_parser
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer

def infer_one_image(model, input, save_dir=None):
    """
    Infer the alpha matte of one image.
    Input:
        model: the trained model
        image: the input image
        trimap: the input trimap
    """
    output = model(input)['phas'].flatten(0, 2)
    output = F.to_pil_image(output)
    output.save(opj(save_dir))

    return None

def init_model(config, checkpoint):
    """
    Initialize the model.
    Input:
        config: the config file of the model
        checkpoint: the checkpoint of the model
    """
    cfg = LazyConfig.load(config)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model.eval()
    DetectionCheckpointer(model).load(checkpoint)

    return model

def get_data(image_dir, trimap_dir):
    """
    Get the data of one image.
    Input:
        image_dir: the directory of the image
        trimap_dir: the directory of the trimap
    """
    image = Image.open(image_dir).convert('RGB')
    image = F.to_tensor(image).unsqueeze(0)
    trimap = Image.open(trimap_dir).convert('L')
    trimap = F.to_tensor(trimap).unsqueeze(0)

    return {
        'image': image,
        'trimap': trimap
    }

if __name__ == '__main__':
    #add argument we need:
    parser = default_argument_parser()
    parser.add_argument('--config-dir', type=str, default='configs/ViTMatte_S_100ep.py')
    parser.add_argument('--checkpoint-dir', type=str, default='path/to/checkpoint')
    parser.add_argument('--image-dir', type=str, default='demo/image.png')
    parser.add_argument('--trimap-dir', type=str, default='demo/trimap.png')
    parser.add_argument('--output-dir', type=str, default='demo/result.png')

    args = parser.parse_args()

    input = get_data(args.image_dir, args.trimap_dir)
    print('Initializing model...Please wait...')
    model = init_model(args.config_dir, args.checkpoint_dir)
    print('Model initialized. Start inferencing...')
    alpha = infer_one_image(model, input, args.output_dir)
    print('Inferencing finished.')