import os
import json
import logging
import argparse
from glob import glob

import numpy as np
import tensorflow as tf

from model import PopMusicTransformer


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logging.basicConfig(
    format='[%(asctime)s | %(levelname)s] %(message)s',
    level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./pretrained/REMI-tempo-checkpoint",
        help="The path to a pretrained checkpoint."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/training",
        help="Path to the training .midi files."
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.midi",
        help="Pattern to look for MIDI files with in `data_dir`."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help=(
            "Path to the output directory."
            " This directory is also used to save checkpoints."
        )
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random number generation seed."
    )
    args = parser.parse_args()
    logging.info(f"The following config was set:\n{json.dumps(vars(args), indent=4)}")

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    # declare model
    logging.info("Starting to initialize model from checkpoint...")
    model = PopMusicTransformer(
        checkpoint=args.checkpoint,
        is_training=True)
    logging.info("The model was successfully initialized...")
    # prepare data
    pattern = f'{args.data_dir}/{args.file_pattern}'
    midi_paths = glob(pattern) # you need to revise it
    logging.info(f"Found {len(midi_paths)} with the pattern: {pattern}")
    logging.info("Starting to prepare data...")
    training_data = model.prepare_data(midi_paths=midi_paths)
    logging.info("Data preparation is done...")

    # check output checkpoint folder
    ####################################
    # if you use "REMI-tempo-chord-checkpoint" for the pre-trained checkpoint
    # please name your output folder as something with "chord"
    # for example: my-love-chord, cute-doggy-chord, ...
    # if use "REMI-tempo-checkpoint"
    # for example: my-love, cute-doggy, ...
    ####################################
    output_checkpoint_folder = args.output_dir # your decision
    if not os.path.exists(output_checkpoint_folder):
        os.mkdir(output_checkpoint_folder)
    
    # finetune
    logging.info("Starting training loop...")
    model.finetune(
        training_data=training_data,
        output_checkpoint_folder=output_checkpoint_folder,
        num_epochs=args.num_epochs)
    logging.info("Training is completed...")


    ####################################
    # after finetuning, please choose which checkpoint you want to try
    # and change the checkpoint names you choose into "model"
    # and copy the "dictionary.pkl" into the your output_checkpoint_folder
    # ***** the same as the content format in "REMI-tempo-checkpoint" *****
    # and then, you can use "main.py" to generate your own music!
    # (do not forget to revise the checkpoint path to your own in "main.py")
    ####################################

    # close
    model.close()

if __name__ == '__main__':
    main()
