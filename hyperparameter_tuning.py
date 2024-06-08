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

import argparse, os, pickle

from src.loadfile import load_dataset
from src.hyperparameter_tuning import hyperparameter_tuning, hyperparameter_tuning_cross_val
from src.model import TWPModel, CPModel


MODEL_OUTPUT_PATH = "./models"
DATASETS_PATH = "./datasets"
PLOTS_PATH = "./plots"

def get_args():
    parser = argparse.ArgumentParser(description="Perform an hyperparameter tuning using Optuna")

    """
    Data handling
    """
    parser.add_argument('--split', default='holdOut', choices=["holdOut","crossVal"],
                        help='Dataset split type (default: holdOut)')
    parser.add_argument('--num-folds', type=int, default=10, help='Number of\
                        folds used to perform cross-validation (default: 10)')
    parser.add_argument('--dataset-name', type=str, default='game_of_thrones',
                        help='dataset name (default: game_of_thrones)')
    parser.add_argument('--window-size', type=int, default=2, help='Window size\
                        used when generating training examples (default: 2)')
    parser.add_argument('--neg-samples', type=int, default=0, help='Number of\
                        negative samples with respect to positive samples (default: 0)')

    """
    Training Hyperparameters
    """
    parser.add_argument('-m', '--mode', default='cbow', choices=["cbow","skipgram"],
                        help='Training model node (default: cbow)')
    parser.add_argument('--optimizer', type=str, default="adam", metavar='O',
                        help='learning rate (default: adam)')


    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Load dataset
    print("Loading dataset...")
    corpus = load_dataset(f"{DATASETS_PATH}/{args.dataset_name}.txt")
    #print(dataset)

    # Load trainer
    print("Loading trainer...")
    if(args.mode == "cbow"):
        trainer = TWPModel
    elif(args.mode == "skipgram"):
        trainer = CPModel
    else:
        raise Exception("Invalid mode")
    
    if args.split == "holdOut":
        # Begin Training!
        best_trials = hyperparameter_tuning(trainer,corpus, args.window_size, output_folder=f"{PLOTS_PATH}/{args.mode}_{args.dataset_name}")

    else:
        # Begin Training!
        best_trials = hyperparameter_tuning_cross_val(trainer,corpus,args.window_size, num_folds=args.num_folds, output_folder=f"{PLOTS_PATH}/{args.mode}_{args.dataset_name}")


    # Show the best trials.
    for trial in best_trials:
        print("Params: ",trial.params, "- Results:",trial.values)