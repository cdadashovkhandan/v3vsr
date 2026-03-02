import argparse
import math
from pathlib import Path
import statistics as stats

from PIL import Image
import numpy as np
from natsort import natsorted
import cv2


def read_y(p):
    im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    y = (65.481 * im[..., 0] + 128.553 * im[..., 1] + 24.966 * im[..., 2] + 16.0) / 255.0
    return y


def psnr_y(p1: Path, p2: Path) -> float:
    y1 = read_y(p1)
    y2 = read_y(p2)
    assert y1.shape == y2.shape, f"Shape mismatch: {y1.shape} vs {y2.shape}"
    mse = np.mean((y1 - y2) ** 2)
    if mse == 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def ssim_y(p1, p2):
    y1 = read_y(p1) * 255.
    y2 = read_y(p2) * 255.

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    y1 = y1.astype(np.float64)
    y2 = y2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(y1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(y2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(y1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(y2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(y1 * y2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def tof_y(gt0, gt1, pred0, pred1):
    """https://github.com/Thmen/EGVSR/blob/master/codes/metrics/metric_calculator.py"""
    gt0 = cv2.imread(str(gt0), cv2.IMREAD_GRAYSCALE)
    gt1 = cv2.imread(str(gt1), cv2.IMREAD_GRAYSCALE)
    pred0 = cv2.imread(str(pred0), cv2.IMREAD_GRAYSCALE)
    pred1 = cv2.imread(str(pred1), cv2.IMREAD_GRAYSCALE)

    # forward flow
    true_OF = cv2.calcOpticalFlowFarneback(
        gt0, gt1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    pred_OF = cv2.calcOpticalFlowFarneback(
        pred0, pred1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # EPE
    diff_OF = true_OF - pred_OF
    tOF = np.mean(np.sqrt(np.sum(diff_OF ** 2, axis=-1)))

    return tOF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--time-scale", type=int, default=8)
    args = ap.parse_args()

    gt_videos = natsorted([p for p in args.gt.iterdir() if p.is_dir() and not p.name.startswith(".")])
    out_videos = natsorted([p for p in args.out.iterdir() if p.is_dir() and not p.name.startswith(".")])
    psnrs, psnrs_center = [], []
    ssims, ssims_center = [], []
    tofs = []

    ts = args.time_scale

    for gv, ov in zip(gt_videos, out_videos):
        if (gv / 'sharp').is_dir():
            gv = gv / 'sharp'
        print(gv, ov)

        gt_frames = sorted([p for p in gv.iterdir() if p.suffix.lower() == ".png"], key=lambda p: int(p.stem))
        out_frames = sorted([p for p in ov.iterdir() if p.suffix.lower() == ".png"], key=lambda p: int(p.stem))
        assert abs(len(gt_frames) - len(out_frames)) <= 8, f"Frame count mismatch: {len(gt_frames)} vs {len(out_frames)}"
        gt_frames = gt_frames[:len(out_frames)]
        assert (len(gt_frames) - 1) % ts == 0

        for i_frame in range(0, len(gt_frames) - 1, ts):
            psnrs.append(stats.mean([
                psnr_y(gt_frames[j], out_frames[j]) for j in range(i_frame, i_frame + ts)
            ]))
            psnrs_center.append(stats.mean([
                psnr_y(gt_frames[j], out_frames[j]) for j in (i_frame, i_frame + (ts // 2))
            ]))
            ssims.append(stats.mean([
                ssim_y(gt_frames[j], out_frames[j]) for j in range(i_frame, i_frame + ts)
            ]))
            ssims_center.append(stats.mean([
                ssim_y(gt_frames[j], out_frames[j]) for j in (i_frame, i_frame + (ts // 2))
            ]))

        for i_frame in range(0, len(gt_frames) - 1):
            tofs.append(tof_y(gt_frames[i_frame], gt_frames[i_frame + 1],
                        out_frames[i_frame], out_frames[i_frame + 1]))

        print(f"PSNR: {sum(psnrs) / len(psnrs):.3f} dB ({len(psnrs)} frames, average)")
        print(f"PSNR: {sum(psnrs_center) / len(psnrs_center):.3f} dB ({len(psnrs)} frames, center)")

        print(f"SSIM: {sum(ssims) / len(ssims):.3f} ({len(ssims)} frames, average)")
        print(f"SSIM: {sum(ssims_center) / len(ssims_center):.3f} ({len(ssims_center)} frames, center)")

        print(f"tOF: {sum(tofs) / len(tofs):.3f}")


if __name__ == "__main__":
    main()
