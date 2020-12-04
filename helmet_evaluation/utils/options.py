# -*- coding: utf-8 -*-
import argparse


def args_parser():
    """ argument parser """
    parser = argparse.ArgumentParser()

    parser.add_argument('--contestant_submitted_file_name', type=str, default="test_pred_simple.json",
                        help="contestant submitted json file name")

    args = parser.parse_args()

    return args
