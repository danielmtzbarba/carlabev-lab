import sys
import warnings

from src.drl_cli import run_train_command

warnings.filterwarnings("ignore")


def main(argv=None):
    return run_train_command(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
