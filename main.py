from typing import Optional


import os
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf

from model import PopMusicTransformer


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def timestamp():
    """Return a timestamp in the '%Y%m%d%H%M%S' format."""
    return str(datetime.now().strftime("%Y%m%d%H%M%S"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pretrained/REMI-tempo-checkpoint",
        help="The path to a pretrained checkpoint."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/",
        help=(
            "The output path of a generated midi file."
            " This can either be a path that leads to an dir or"
            " a complete path with filename and the '.midi' suffix."
            " If only a directory is provided, the resulting filename will be a timestamp."
        )
    )
    parser.add_argument(
        "--prompt",
        type=Optional[str],
        default=None,
        help=(
            "A path to a .midi file that shall be continued by the model."
            " By default this evaluates to None"
            " - meaning the music is to be generated from scratch."
        )
    )
    parser.add_argument(
        "--n_target_bar",
        type=int,
        default=16,
        help="Number of bars to be generated."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.2,
        help=(
            "Temperature parameter during sampling."
            " A higher temperature leads to more random generation results."
        )
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help=(
            "Topk sampling parameter."
            " Filters the top k predictions during each autoregressive step."
            " A random option is then picked among them."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random number generation seed."
    )
    
    args = parser.parse_args()
    args = vars(args)

    if os.path.isdir(args["output_path"]):
        args["output_path"] = f"{args['output_path']}/{timestamp()}.midi"

    if args["seed"] is not None:
        np.random.seed(args["seed"])
        tf.random.set_seed(args["seed"])

    model = PopMusicTransformer(
        checkpoint=args["checkpoint"],
        is_training=False)
    
    model.generate(
        n_target_bar=args["n_target_bar"],
        temperature=args["temperature"],
        topk=args["topk"],
        output_path=args["output_path"],
        prompt=args["prompt"])
    
    model.close()


if __name__ == '__main__':
    main()
