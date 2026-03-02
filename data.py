import random
from pathlib import Path
from typing import NamedTuple
from functools import cache
import math

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as f
from torchvision.transforms import RandomCrop
from torchvision.datasets.folder import IMG_EXTENSIONS
from PIL import Image

from utils import make_grid, pil_resize

INTERP_MODE = 'bicubic'
TARGET_SIZE = 48


def to_uint8_tensor(image: Image):
    return torch.from_numpy(np.array(image)).permute(2, 0, 1)


def open_img(path):
    with open(str(path), "rb") as f:
        img = Image.open(f).convert("RGB")
        img = to_uint8_tensor(img)
    return img


class DataShard(NamedTuple):
    rank: int
    world_size: int


def nearest_interp1d(x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    indices = coords.round().long().clamp(0, x.size(0) - 1)
    return x[indices]


@cache
def make_z_coords(every: int, seq_len: int):
    lr_coords = torch.linspace(-0.5, 0.5, seq_len, dtype=torch.float64)
    hr_coords = torch.linspace(-0.5, 0.5, (seq_len - 1) * every + 1, dtype=torch.float64)

    hr_coords_int = (hr_coords + 0.5) * (seq_len - 1)
    hr_coords_int -= 1e-5  # makes sure the center frame belongs to the previous input frame
    interp_coords = nearest_interp1d(lr_coords, hr_coords_int)
    rel_coords = (hr_coords - interp_coords) * (seq_len - 1)

    return rel_coords.float()


def sample_z_indices(every: int, seq_len: int):
    """Stratified sampling of indices for target frames, one per input frame."""
    indices, current = [], 0
    for i in range(seq_len):
        block_len = every
        if i == 0:
            block_len = every // 2 + 1
        elif i == seq_len - 1:
            block_len = math.ceil(every / 2)
        indices.append(current + random.randrange(0, block_len))
        current += block_len
    return indices


class REDSVideoFolder(Dataset):

    def __init__(self, root: Path, seq_len: int, shard: DataShard, every: list[int],
                 no_time_jitter=False):
        if (root / 'GT').exists():
            root = root / 'GT'
        self.seq_len = seq_len
        self.every = every
        self.no_time_jitter = no_time_jitter

        video_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        video_dirs = video_dirs[shard.rank::shard.world_size]
        self.items = []
        for video_dir in video_dirs:
            video = []
            for path in sorted(list(video_dir.rglob('*'))):
                if path.suffix in IMG_EXTENSIONS:
                    video.append(path)
            self.items.append(video)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        items = self.items[i]

        every = random.choice(self.every)
        need = (self.seq_len - 1) * every + 1
        assert need <= len(items)
        t_start = random.randint(0, len(items) - need)
        items = items[t_start:t_start + need]

        if random.random() < 0.5:
            # random temporal flip
            items = items[::-1]

        def maybe_load(obj):
            return obj if torch.is_tensor(obj) else open_img(obj)

        input_frames = torch.stack([maybe_load(p) for p in items[::every]])
        assert len(input_frames) == self.seq_len

        if self.no_time_jitter:
            target_frames_idc = list(range(0, need, every))
        else:
            target_frames_idc = sample_z_indices(every, self.seq_len)
        target_frames = torch.stack([maybe_load(items[i]) for i in target_frames_idc])
        z_coords = make_z_coords(every, self.seq_len)
        target_frames_coords = z_coords[target_frames_idc]
        assert (not self.no_time_jitter or
                torch.allclose(target_frames_coords, torch.zeros_like(target_frames_coords), atol=1e-6))

        return input_frames, target_frames, target_frames_coords


def read_video(path) -> list[torch.Tensor]:
    path = Path(path)
    frames = []
    capture = cv2.VideoCapture(str(path))

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1)
        frames.append(frame)

    capture.release()
    return frames


class Adobe240VideoFolder(REDSVideoFolder):

    def __init__(self, root: Path, seq_len: int, shard: DataShard, every: list[int],
                 no_time_jitter=False):
        assert every == [8]  # canonical setting

        self.seq_len = seq_len
        self.every = every
        self.no_time_jitter = no_time_jitter

        video_paths = sorted([n for n in root.iterdir()])
        video_paths = video_paths[shard.rank::shard.world_size]

        self.items = []
        for p in video_paths:
            frames = read_video(p)
            self.items.append(frames)

        print(f'Read Adobe dataset with {len(self)} videos.')


class REDSEvalVideoFolder(Dataset):
    """REDS VideoFolder for eval, returning full videos instead of windows"""

    def __init__(self, root: Path, shard):
        if (root / 'GT').exists():
            root = root / 'GT'

        video_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        video_dirs = video_dirs[shard.rank::shard.world_size]
        self.items = []
        for video_dir in video_dirs:
            video = []
            for path in sorted(list(video_dir.rglob('*'))):
                if path.suffix in IMG_EXTENSIONS:
                    video.append(path)
            self.items.append(video)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        paths = self.items[i]
        video = [open_img(p) for p in paths]
        return torch.stack(video)


class HFEvalVideoFolder(Dataset):
    """Adobe240 VideoFolder for eval, returning full videos instead of windows"""

    def __init__(self, root: Path, shard):
        video_paths = sorted([n for n in root.iterdir()])
        video_paths = video_paths[shard.rank::shard.world_size]
        self.paths = video_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        if path.is_file():
            frames = read_video(self.paths[i])
        elif path.is_dir() and not path.name.startswith('.'):
            frames = []
            for path in sorted(list(path.glob('*.png'))):
                frames.append(open_img(path))
        else:
            raise FileNotFoundError

        return torch.stack(frames)


class ContinuousWrapper(Dataset):

    def __init__(
            self,
            dataset,
            patch_size: int,
            seq_len: int,
            *,
            scale_range: tuple[float, float],
            augment_scale_range: tuple[float, float],
            augment_scale_prob: float,
            transforms):
        self.dataset = dataset
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.scale_range = scale_range
        self.augment_scale_range = augment_scale_range
        self.augment_scale_prob = augment_scale_prob
        self.transforms = transforms

        assert augment_scale_range[0] >= 1

        # helper tensor for temporal indexing
        self.t_idc = torch.arange(seq_len)[:, None].repeat(1, TARGET_SIZE ** 2)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        input_frames, target_frames, target_frames_coords = self.dataset[item]

        scale = random.uniform(*self.scale_range)
        augment_scale = random.uniform(*self.augment_scale_range) \
            if random.random() < self.augment_scale_prob else 1

        all_frames = torch.cat([input_frames, target_frames])
        all_frames = RandomCrop(round(self.patch_size * scale * augment_scale))(all_frames)
        all_frames = all_frames.float() / 255.
        target_size = int(all_frames.shape[-1] / augment_scale)
        if augment_scale != 1:
            all_frames = torch.stack([pil_resize(t, (target_size, target_size)) for t in all_frames])
        all_frames = self.transforms(all_frames)
        input_frames, target_frames = all_frames.chunk(2)

        source = torch.stack([pil_resize(t, (self.patch_size, self.patch_size)) for t in input_frames])
        del input_frames
        source_up = f.interpolate(source, (target_size, target_size), mode='nearest-exact')
        effective_scale = target_size / self.patch_size

        source, source_up, target = \
            source.permute(0, 2, 3, 1), source_up.permute(0, 2, 3, 1), target_frames.permute(0, 2, 3, 1)

        # get spatially random pixel samples (temporal sampling has been done by VideoDataset)
        target_coords = (torch.from_numpy(make_grid(target_size)).view(-1, 2)
                         .repeat(target.shape[0], 1, 1))
        target_flat = target.reshape(target.shape[0], -1, 3)
        source_up_flat = source_up.reshape(target.shape[0], -1, 3)
        sample_idc = torch.stack([torch.randperm(target_size**2)[:TARGET_SIZE**2]
                                  for _ in range(target.shape[0])])
        target_coords = target_coords[self.t_idc, sample_idc]
        target_flat = target_flat[self.t_idc, sample_idc]
        source_up_flat = source_up_flat[self.t_idc, sample_idc]

        return {
            'source': source,
            'target_coords': target_coords.view(target.shape[0], TARGET_SIZE, TARGET_SIZE, -1),
            'target_coords_z': target_frames_coords,
            'target': target_flat.view(target.shape[0], TARGET_SIZE, TARGET_SIZE, -1),
            'source_nearest': source_up_flat.view(target.shape[0], TARGET_SIZE, TARGET_SIZE, -1),
            'scale': effective_scale
        }
