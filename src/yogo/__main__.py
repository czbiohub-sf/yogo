import sys
import torch

from yogo.infer import do_infer
from yogo.train import do_training
from yogo.argparsers import global_parser

no_onnx = False
onnx_err = ""
try:
    from yogo.export_model import do_export
except ImportError as e:
    no_onnx = True
    onnx_err = str(e)


def main():
    print('cudaver', torch.version.cuda)
    p = global_parser()
    args = p.parse_args()

    if args.task == "train":
        do_training(args)
    elif args.task == "export":
        if no_onnx:
            print("onnx is not installed; install yogo with `pip3 install yogo[onnx]`")
            print(f"recieved error {onnx_err}")
            sys.exit(1)
        do_export(args)
    elif args.task == "infer":
        do_infer(args)
    else:
        p.print_help()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
