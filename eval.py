import sys

from src.drl_cli import run_eval_command


def main(argv=None):
    return run_eval_command(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
