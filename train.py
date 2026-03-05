import os
import torch

import warnings

from src.config.experiment_loader import load_experiment, run_experiment

warnings.filterwarnings("ignore")

def main():
    cfg = load_experiment()
    run_experiment(cfg)

if __name__ == "__main__":
    main()
