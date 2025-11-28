import os
from data import common
import numpy as np
from PIL import Image
import torch.utils.data as data

class FMDTest(data.Dataset):
    def __init__(self, args, train=False):
        self.args = args
        self.train = train
        self.name = 'FMDTest'
        self.noise_g = args.noise_g
        self.idx_scale = 0
        self.benchmark = False


        base = "path/for/test/files"
        self.clean_dir = os.path.join(base, "gt")
        self.noisy_dir = os.path.join(base, "raw")

        print("Clean dir:", self.clean_dir)
        print("Noisy dir:", self.noisy_dir)

        clean_files = sorted([f for f in os.listdir(self.clean_dir) if f.endswith(".png")])
        noisy_files = sorted([f for f in os.listdir(self.noisy_dir) if f.endswith(".png")])

        print(f"Found {len(clean_files)} clean images")
        print(f"Found {len(noisy_files)} noisy images")

        self.filelist = []
        for clean_f in clean_files:
            noisy_path = os.path.join(self.noisy_dir, clean_f)
            clean_path = os.path.join(self.clean_dir, clean_f)
            if os.path.exists(noisy_path):
                self.filelist.append((noisy_path, clean_path))
            else:
                print(f"No noisy version found for: {clean_f}")

        print(len(self.filelist), "image pairs")

    def __getitem__(self, idx):
        noisy_path, clean_path = self.filelist[idx]

        lr = np.array(Image.open(noisy_path).convert("L"))
        hr = np.array(Image.open(clean_path).convert("L"))

        lr = np.stack([lr, lr, lr], axis=-1)
        hr = np.stack([hr, hr, hr], axis=-1)

        lr = common.set_channel([lr], self.args.n_colors)[0]
        hr = common.set_channel([hr], self.args.n_colors)[0]

        lr_tensor = common.np2Tensor([lr], self.args.rgb_range)[0]
        hr_tensor = common.np2Tensor([hr], self.args.rgb_range)[0]

        filename = os.path.splitext(os.path.basename(noisy_path))[0]

        return lr_tensor, hr_tensor, filename, 0

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale