from yolox.core import TrainerV2
from yolox.exp import get_exp_v2

import argparse


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n",
                        "--name",
                        type=str,
                        default=None,
                        help="model name")

    parser.add_argument("-b",
                        "--batch-size",
                        type=int,
                        default=16,
                        help="batch size")
    parser.add_argument("-d",
                        "--devices",
                        default=None,
                        type=int,
                        help="device for training")
    parser.add_argument("--local_rank",
                        default=0,
                        type=int,
                        help="local rank for dist training")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your expriment description file",
    )
    parser.add_argument("--resume",
                        default=False,
                        action="store_true",
                        help="resume training")
    parser.add_argument("-c",
                        "--ckpt",
                        default=None,
                        type=str,
                        help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        required=True,
        help="input width of model, will override exp if set",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        required=True,
        help="input height of model, will override exp if set",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )

    return parser


def main(exp, args):

    trainer = TrainerV2(exp, args)
    trainer.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp_v2(args.exp_file,
                     args.name,
                     input_width=args.input_width,
                     input_height=args.input_height)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    main(exp, args)
