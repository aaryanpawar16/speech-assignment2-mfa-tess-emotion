"""Train the fusion pipeline.   python models/fusion_pipeline/train.py --help"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.cli import train_main  # noqa: E402

if __name__ == "__main__":
    train_main("fusion")
