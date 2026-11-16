"""Test the text pipeline.   python models/text_pipeline/test.py --help"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.cli import test_main  # noqa: E402

if __name__ == "__main__":
    test_main("text")
