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

import os

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optuna.visualization import plot_contour, plot_parallel_coordinate, plot_param_importances, plot_rank

from src.preprocess import create_training_sets

def hyperparameter_tuning(trainer_class, corpus, window_size, output_folder):
    def objective(trial):
        params = {
            'window_size': window_size,
            'vocab_size': 1000000,
            'embedding_len': trial.suggest_int("embedding_len", 8, 512),
            'num_dense': trial.suggest_int("num_dense", 1, 3),
            'dropout_rate': trial.suggest_float("dropout_rate", 0.0, 1.0),
            'num_units': trial.suggest_int("num_units", 16, 128),
            'epochs': 10,
            'batch_size': 128,
            'lr': 0.1,
            'optimizer': "adam",
            'seed': 42,
            'neg_samples':10
        }

        # Create model
        trainer = trainer_class(params)

        # Compute vocabulary
        trainer.compute_vocabulary(corpus)

        # Prepare model
        trainer.build_model()

        # Prepare dataset
        X, y = trainer.preprocess_data(corpus)
        train_ds, val_ds = create_training_sets(X,y)

        # Begin Training!
        history = trainer.train(train_ds,val_ds)
        return max(history.history['val_accuracy'])
    
    study = optuna.create_study(
        directions=["maximize"]
    )
    study.optimize(objective, callbacks=[MaxTrialsCallback(5, states=(TrialState.COMPLETE,))])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plot_parallel_coordinate(study, target=lambda t: t.values[0], target_name="Accuracy").write_image(file=f'{output_folder}/parallel_coordinate.png', format='png')
    plot_param_importances(study, target=lambda t: t.values[0], target_name="Accuracy").write_image(file=f'{output_folder}/param_importances.png', format='png')
    plot_contour(study, target=lambda t: t.values[0], target_name="Accuracy").write_image(file=f'{output_folder}/contour.png', format='png')
    plot_rank(study, target=lambda t: t.values[0], target_name="Accuracy").write_image(file=f'{output_folder}/rank.png', format='png')

    # Show the best trials.
    return study.best_trials

def hyperparameter_tuning_cross_val(trainer_class, corpus, window_size, num_folds, output_folder):
    def objective(trial):
        params = {
            'window_size': window_size,
            'vocab_size': 1000000,
            'embedding_len': trial.suggest_int("embedding_len", 8, 512),
            'num_dense': trial.suggest_int("num_dense", 1, 3),
            'dropout_rate': trial.suggest_float("dropout_rate", 0.0, 1.0),
            'num_units': trial.suggest_int("num_units", 16, 128),
            'epochs': 10,
            'batch_size': 128,
            'optimizer': "adam",
            'seed': 42,
            'neg_samples':10
        }

        # Create model
        trainer = trainer_class(params)

        # Compute vocabulary
        trainer.compute_vocabulary(corpus)

        # Prepare model
        trainer.build_model()

        # Prepare dataset
        X, y = trainer.preprocess_data(corpus)

        # Begin Training!
        _, metric = trainer.cross_val_train((X,y), num_folds=num_folds)
        return metric
    
    study = optuna.create_study(
        directions=["maximize"]
    )
    study.optimize(objective, callbacks=[MaxTrialsCallback(10, states=(TrialState.COMPLETE,))])

    # Plot results
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plot_parallel_coordinate(study, target=lambda t: t.values[0], target_name="Accuracy").write_image(file=f'{output_folder}/parallel_coordinate.png', format='png')
    plot_param_importances(study, target=lambda t: t.values[0], target_name="Accuracy").write_image(file=f'{output_folder}/param_importances.png', format='png')
    plot_contour(study, target=lambda t: t.values[0], target_name="Accuracy").write_image(file=f'{output_folder}/contour.png', format='png')
    plot_rank(study, target=lambda t: t.values[0], target_name="Accuracy").write_image(file=f'{output_folder}/rank.png', format='png')

    # Show the best trials.
    return study.best_trials