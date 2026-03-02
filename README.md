# Continuous Space-Time Video Super-Resolution with 3D Fourier Fields

**ICLR 2026 (accepted)**

[![Page](https://img.shields.io/badge/Project-Page-green)](https://v3vsr.github.io)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2509.26325)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

This is the official implementation of the paper
> *Continuous Space-Time Video Super-Resolution with 3D Fourier Fields* <br>
> [Alexander Becker](https://a1ex.dev),
[Julius Erbach](https://juliuserbach.github.io),
[Dominik Narnhofer](https://scholar.google.com/citations?user=tFx8AhkAAAAJ&hl=en),
[Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en) <br>
> ETH Zurich

## News
2026-01-26: Paper is accepted to ICLR 2026. 🎉 <br>
2025-09-30: Paper is now on [arXiv](https://arxiv.org/abs/2509.26325)).<br>

## Setup

This code is tested on Linux with Python 3.11 and an NVIDIA GPU with > 20GB VRAM (e.g., {3,4,5}090 or H100). Install the required packages with
```bash
pip install -r requirements.txt
```

Then build deformable attention CUDA extensions for JAX:
```bash
cd models/rvrt_jax/deform_attn && ./build.sh && cd -
```

## Evaluate pre-trained models

Download pre-trained V3 checkpoint from [Google Drive](https://drive.google.com/file/d/1V7LtyhHZdnw2TOpJGl3GwKY8ksgL0ldb/view?usp=share_link) and place them in the `checkpoints/` folder.

An example command for evaluating on the Vid4 dataset (x2 temporal upsampling) is:
```bash
python run_inference.py --data-dir data/ --checkpoint-path checkpoints/v3.pkl --eval-sets Vid4 --space-scale 4 --time-scale 2
```
Make sure that `--data-dir` points to a folder containing the dataset (i.e., in this case with a subfolder `Vid4`), and that `--checkpoint-path` points to a downloaded pre-trained model. For the default Adobe240 and GoPro benchmarks you likely want to use `--space-scale 4 --time-scale 8`.

We have included some logic for quickly evaluating the relevant metrics of the inference results against the ground truth, e.g.,
```bash
python eval_imgs.py data/Vid4 results/Vid4/x4 --time-scale 2
```

## Train your own model

To train on Adobe240, download the dataset from the official sources and then place videos under ./data with the following structure:

```
data/
├── Adobe240
│   ├── train
│   │   ├── ...
│   ├── valid
│   │   ├── ...
│   ├── test
│   │   ├── ...
```

Then use the convenience training script:
```bash
bash run_train.sh
```
Make sure to check `run_train.sh` and `args.py` for the availabe training options.

## Citation
```bibtex
@article{becker2025continuous,
  title={Continuous Space-Time Video Super-Resolution with 3D Fourier Fields},
  author={Becker, Alexander and Erbach, Julius and Narnhofer, Dominik and Schindler, Konrad},
  journal={arXiv preprint arXiv:2509.26325},
  year={2025}
}
```
