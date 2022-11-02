import sys

from yogo.train import do_training
from yogo.argparsers import global_parser

no_onnx = False
try:
    from yogo.export_model import do_export
except ImportError:
    no_onnx = True


def main():
    p = global_parser()
    args = p.parse_args()

    if args.task == "train":
        do_training(args)
    elif args.task == "export":
        if no_onnx:
            print("onnx is not installed; install yogo with `pip3 install yogo[onnx]`")
            sys.exit(1)
        do_export(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
