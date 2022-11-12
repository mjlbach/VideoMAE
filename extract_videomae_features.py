# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from pathlib import Path
from timm.models import create_model
import h5py
import utils
import modeling_pretrain
import tqdm
from einops import rearrange
from torchvision import transforms
from transforms import *
from masking_generator import  TubeMaskingGenerator

class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406] # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225] # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupCenterCrop(args.input_size)
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data , _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

def get_args():
    parser = argparse.ArgumentParser('VideoMAE visualization reconstruction script', add_help=False)
    parser.add_argument('--model_path', type=str, help='checkpoint path of model')
    parser.add_argument('--mask_type', default='random', choices=['random', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    parser.add_argument(
        "--trajectory_path", type=str, help="input video path", default=None
    )
    parser.add_argument("--dataset_path", type=str, help="input video path", default=None)
    return parser.parse_args()


def get_model(args):
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth
    )

    return model

def process_sequence(args, model, device, path):
    f = h5py.File(path.joinpath("data.hdf5"), "r+")
    if "videomae_features" in f:
        f.close()
        return
    else:
        f.close()

    img_arr = []
   
    for img in map(str, path.glob("*.png")):
        img_arr.append(np.array(Image.open(img)))
    vr = np.stack(img_arr)
    num_frames = vr.shape[0]
    pad = opts.num_frames // 2
    vr = np.pad(vr, [(pad, pad), (0,0), (0,0), (0,0)]) #type: ignore

    img_arr = np.array([Image.fromarray(vid).convert('RGB') for vid in vr])
    patch_size = args.patch_size
    transforms = DataAugmentationForVideoMAE(args)
    embedding_arr = torch.zeros(num_frames, 15360)
    for idx in range(num_frames):
        img = img_arr[np.arange(0, 16) + idx]

        # frame_id_list = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61]
        img, bool_masked_pos = transforms((img, None)) # T*C,H,W

        img = img.view((args.num_frames , 3) + img.size()[-2:]).transpose(0,1) # T*C,H,W -> T,C,H,W -> C,T,H,W
        # img = img.view(( -1 , args.num_frames) + img.size()[-2:]) 
        bool_masked_pos = torch.from_numpy(bool_masked_pos)

        with torch.no_grad():
            # img = img[None, :]
            # bool_masked_pos = bool_masked_pos[None, :]
            img = img.unsqueeze(0)
            bool_masked_pos = bool_masked_pos.unsqueeze(0)
            
            img = img.to(device, non_blocking=True)
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
            features = {}
            def get_features(name):
                def hook(model, input, output): # type: ignore
                    try:
                        features[name] = {
                            # "input": input,
                            "output": output
                        }
                    except:
                        breakpoint()
                return hook
            # model.encoder_to_decoder.register_forward_hook(get_features('encoder_to_decoder'))
            model.encoder.register_forward_hook(get_features('encoder'))
            model(img, bool_masked_pos)

            new_features = features['encoder']['output']
            correspondance = torch.zeros_like(img)
            for frame in range(correspondance.shape[2]):
                correspondance[:, :, frame] = frame

            correspondance = rearrange(correspondance, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size[0], p2=patch_size[0])
            correspondance = correspondance[~bool_masked_pos]
            true_tiles = correspondance[:, 0, 0]
            frame_to_extract = 8
            embedding_arr[idx] = new_features[:, true_tiles == frame_to_extract].reshape(-1)

    f = h5py.File(path.joinpath("data.hdf5"), "r+")
    f.create_dataset(f"videomae_features", data=np.array(embedding_arr.cpu()))
    f.close()

def main(args):
    print(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    assert (
        opts.trajectory_path or opts.dataset_path
    ), "Either --trajectory_path or --dataset_path must be passed to the script"

    if args.trajectory_path:
        process_sequence(args, model, device, Path(args.trajectory_path))
    elif args.dataset_path:
        pbar = tqdm.tqdm(list(Path(args.dataset_path).rglob("data.hdf5")))

        for path in pbar:
            pbar.set_description(f"{path}")  # type: ignore
            process_sequence(args, model, device, path.parent)

if __name__ == '__main__':
    opts = get_args()
    main(opts)
