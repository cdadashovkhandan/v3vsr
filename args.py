import configargparse
import numpy as np

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, type=str)

# training
parser.add_argument('--n-iter', type=int, default=2_500_000)
parser.add_argument('--local-batch-size', type=int, default=1)
parser.add_argument('--val-every', type=int, default=5_000)
parser.add_argument('--val-samples', type=int, default=256)
parser.add_argument('--patch-size', type=int, default=80, help='Image size at t=1.0')
parser.add_argument('--seq-len', type=int, default=13, help='Sequence length of patches')
parser.add_argument('--scale-range', type=float, nargs='+', default=(1.2, 4.))
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--num-workers', type=int, default=30)
parser.add_argument('--loss', type=str, default='MAE')
parser.add_argument('--seed', type=int, default=2218)
parser.add_argument('--max-grad-norm', type=float, default=1.)
parser.add_argument('--augment-scale-range', type=float, nargs='+', default=(1., 2.0))
parser.add_argument('--augment-scale-prob', type=float, default=0.5)
parser.add_argument('--accu-steps', type=int, default=1)
parser.add_argument('--pretrained-raft', type=str, default=None)
parser.add_argument('--pretrained-encoder', type=str, default=None)
parser.add_argument('--freeze-encoder-first', type=int, default=0)
parser.add_argument('--encoder-grad-multiplier', type=float, default=1.)
parser.add_argument('--freeze-flow-first', type=int, default=2_200_000)
parser.add_argument('--flow-grad-multiplier', type=float, default=.1)
parser.add_argument('--t-init-scale', type=float, default=1.)
parser.add_argument('--mp-policy', default='params=float32,compute=float16,output=float32')
parser.add_argument('--every-frame', type=int, nargs='+', default=[8])

# model
parser.add_argument('--num-basis', type=int, default=512)
parser.add_argument('--k', type=float, default=np.sqrt(np.log(4)) / (np.pi ** 2 * 2))
parser.add_argument('--init-scale', type=float, default=16.0)
parser.add_argument('--embed-dims', type=int, nargs=3, default=[90, 90, 90])
parser.add_argument('--num-blocks', type=int, nargs=3, default=[2, 4, 2])
parser.add_argument('--depths', type=int, nargs=3, default=[2, 2, 2])
parser.add_argument('--attention-heads', type=int, default=12)
parser.add_argument('--deformable-groups', type=int, default=12)
parser.add_argument('--output-dims', type=int, default=256)
parser.add_argument('--use-remat', action='store_true')
parser.add_argument('--raft-size', default='large')

# data
parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--train-set', type=str, default='REDS_train')
parser.add_argument('--val-set', type=str, default='REDS_val')
parser.add_argument('--no-wandb', action='store_true')
parser.add_argument('--wandb-project', type=str, default='avsr')
parser.add_argument('--wandb-dir', type=str, default='../logs')
parser.add_argument('--tag', type=str, default='', help='Tag to append to checkpoint file name')

# for evaluation only
parser.add_argument('--save-dir', type=str, default='./results')
parser.add_argument('--eval-sets', type=str, nargs='+', default=['REDS_val'])
parser.add_argument('--checkpoint-path', type=str)
parser.add_argument('--space-scale', type=int, default=4)
parser.add_argument('--time-scale', type=int, default=8)
parser.add_argument('--geo-flip', action='store_true', help='Add flips to geo-ensemble')
parser.add_argument('--y-only', action='store_true', help='Only evaluate Y channel of YCbCr image')

parser.add_argument('--thera_dim', type=int, default=512, help='thera dimension (I have no idea)')
