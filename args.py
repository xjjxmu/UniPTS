import argparse
import sys
import yaml

from configs import parser as _parser

args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training for STR, DNW and GMP")

    # General Config
    parser.add_argument(
        "--data", help="path to dataset base directory", default="/path/to/imagenet/"
    )
    parser.add_argument(
    '--num_samples_for_calib',
    type=int,
    default = 10240,
    help='image size for finetune. default:10240'
    )
    parser.add_argument(
    '--num_samples_for_bn',
    type=int,
    default = 10240,
    help='image size for calibrating BN. default:10240'
    )
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument("--set", help="name of dataset", type=str, default="ImageNet")
    parser.add_argument(
        "-a", "--arch", metavar="ARCH", default="ResNet18", help="model architecture"
    )
    parser.add_argument("--layerwise", help="layerwisesparsity", type=str, default="mask")
    parser.add_argument("--sparse_schedule", help="sparse_schedule", type=str, default="1")
    parser.add_argument(
        "--config", help="Config file to use (see configs dir)", default=None
    )
    parser.add_argument(
        "--log-dir", help="Where to save the runs. If None use ./runs", default=None
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=20,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 20)",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=None,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--iters",
        default=4000,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=64,
        type=int,
        metavar="N",
        help="mini-batch isize (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.001,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--mask_lr",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--warmup_length", default=20, type=int, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--decay_epochs", default=30, type=int, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--min_lr", default=0.0, type=float, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--warmup_lr", default=0.0, type=float, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--decay_rate", default=0.1, type=float, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=1000,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--num-classes",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
    '--checkpoint',
    type=str,
    default='/path/to/dense_model/',
    help='pretrain checkpoint direction'
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--evaluate_model_link",
        dest="evaluate",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seed", default=1, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which GPUs to use for multigpu training",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which GPUs to use ",
    )
    parser.add_argument(
        "--lr-policy", default="constant_lr", help="Policy for the learning rate."
    )
    parser.add_argument(
        "--multistep-lr-adjust", default=30, type=int, help="Interval to drop lr"
    )
    parser.add_argument(
        "--multistep-lr-gamma", default=0.1, type=int, help="Multistep multiplier"
    )
    parser.add_argument(
        "--name", default=None, type=str, help="Experiment name to append to filepath"
    )
    parser.add_argument(
        "--save_every", default=20, type=int, help="Save every ___ epochs"
    )
    parser.add_argument(
        "--prune_rate",
        default=0.0,
        help="Amount of pruning to do during sparse training",
        type=float,
    )
    parser.add_argument(
        "--width-mult",
        default=1.0,
        help="How much to vary the width of the network.",
        type=float,
    )
    parser.add_argument(
        "--nesterov",
        default=False,
        action="store_true",
        help="Whether or not to use nesterov for SGD",
    )
    parser.add_argument(
        "--one-batch",
        action="store_true",
        help="One batch train set for debugging purposes (test overfitting)",
    )
    parser.add_argument(
        "--conv-type", type=str, default=None, help="What kind of sparsity to use"
    )
    parser.add_argument(
        "--freeze-weights",
        action="store_true",
        help="Whether or not to train only mask (this freezes weights)",
    )
    parser.add_argument(
        "--freeze-masks",
        action="store_true",
        help="Whether or not to train only weight (this freezes masks)",
    )
    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
    )
    parser.add_argument("--bn-type", default=None, help="BatchNorm type")
    parser.add_argument(
        "--init", default="kaiming_normal", help="Weight initialization modifications"
    )
    parser.add_argument(
        "--no-bn-decay", action="store_true", default=False, help="No batchnorm decay"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="debug"
    )
    parser.add_argument(
        "--dense-conv-model", action="store_true", default=False, help="Store a model variant of the given pretrained model that is compatible to CNNs with DenseConv (nn.Conv2d)"
    )
    parser.add_argument(
        "--st-decay", type=float, default=None, help="decay for sparse thresh. If none then use normal weight decay."
    )

    parser.add_argument(
        "--scale-fan", action="store_true", default=False, help="scale fan"
    )
    parser.add_argument(
        "--first-layer-dense", action="store_true", help="First layer dense or sparse"
    )
    parser.add_argument(
        "--last-layer-dense", action="store_true", help="Last layer dense or sparse"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        help="Label smoothing to use, default 0.0",
        default=None,
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="sigmoid",
        default=1,
    )
    parser.add_argument(
        "--first-layer-type", type=str, default=None, help="Conv type of first layer"
    )
    parser.add_argument(
        "--sInit-type",
        type=str,
        help="type of sInit",
        default="constant",
    )
    parser.add_argument(
        "--sInit-value",
        type=float,
        help="initial value for sInit",
        default=100,
    )
    parser.add_argument(
        "--sparse-function", type=str, default='sigmoid', help="choice of g(s)"
    )
    parser.add_argument(
        "--use-budget", action="store_true", help="use the budget from the pretrained model."
    )
    parser.add_argument(
        "--ignore-pretrained-weights", action="store_true", help="ignore the weights of a pretrained model."
    )
    parser.add_argument(
        "--print_freq", type=int, default=10, help="print frequency"
    )
    args = parser.parse_args()

    get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
