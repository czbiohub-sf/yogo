from yogo.train import do_training
from yogo.export_model import do_export
from yogo.argparsers import global_parser


def main():
    p = global_parser()
    args = p.parse_args()

    if args.task == "export":
        do_export(args)
    elif args.task == "train":
        do_training(args)


if __name__ == "__main__":
    main()
