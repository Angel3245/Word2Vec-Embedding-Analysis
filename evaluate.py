# Copyright (C) 2024  Jose Ángel Pérez Garrido
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse, pickle
from src.loadfile import load_dataset

DATASETS_PATH = "./datasets"
MODEL_OUTPUT_PATH = "./models"

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a previously trained Word2vec model")

    """
    Data handling
    """
    parser.add_argument('--dataset-name', type=str, default='game_of_thrones',
                        help='dataset name (default: game_of_thrones)')
    parser.add_argument('--window-size', type=int, default=2, help='Window size\
                        used when generating training examples (default: 2)')
    parser.add_argument('--neg-samples', type=int, default=0, help='Number of\
                        negative samples with respect to positive samples (default: 0)')

    """
    Model Parameters
    """
    parser.add_argument('--vocab-size', type=int, default=1000000, help='Vocabulary\
                        size (default: 1000000)')

    """
    Hyperparameters
    """
    parser.add_argument('-m', '--mode', default='cbow', choices=["cbow","skipgram"],
                        help='Training model node (default: cbow)')
    parser.add_argument('--batch-size', type=int, default=128,
                        metavar='N', help='number of examples in a batch (default: 128)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')



    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Load dataset
    print("Loading dataset...")
    corpus = load_dataset(f"{DATASETS_PATH}/{args.dataset_name}.txt")

    # Load model
    print("Loading pickle model...")
    with open(str(f"{MODEL_OUTPUT_PATH}/{args.mode}_{args.dataset_name}.pickle"), "rb") as data_file:
        trainer = pickle.load(data_file)

    print("Preprocessing dataset...")
    # Preprocess data
    X, y = trainer.preprocess_data(corpus)

    # Evaluate model
    print("[Loss, Accuracy] -",trainer.evaluate((X,y)))