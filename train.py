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
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorboard.plugins import projector

from src.model import TWPModel, CPModel, CPModel_Tokenizer
from src.loadfile import load_dataset
from src.preprocess import standardization, create_training_sets
from src.embeddings import visualize_tsne_embeddings

MODEL_OUTPUT_PATH = "./models"
DATASETS_PATH = "./datasets"
PLOTS_PATH = "./plots"
TARGET_WORDS_PATH = "./target_words"

def plot_training(history, model_name):
    # Generate training plots
    print("Generating training plots...")
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
    #plt.show()

    # Save plot
    if not os.path.exists(str(f"{PLOTS_PATH}/{model_name}")):
        os.makedirs(str(f"{PLOTS_PATH}/{model_name}"))
    plt.savefig(str(f"{PLOTS_PATH}/{model_name}/Plot_accuracy.png"))
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left')
    #plt.show()

    # Save plot
    plt.savefig(str(f"{PLOTS_PATH}/{model_name}/Plot_loss.png"))
    plt.close()

def get_args():
    parser = argparse.ArgumentParser(description="Train a word2vec model")

    """
    Data handling
    """
    parser.add_argument('--split', default='none', choices=["holdOut","none"],
                        help='Dataset split type (default: none)')
    parser.add_argument('--num-folds', type=int, default=2, help='Number of\
                        folds used to perform cross-validation (default: 10)')
    parser.add_argument('--dataset-name', type=str, default='game_of_thrones',
                        help='dataset name (default: game_of_thrones)')
    parser.add_argument('--window-size', type=int, default=2, help='Window size\
                        used when generating training examples (default: 2)')
    parser.add_argument('--neg-samples', type=int, default=0, help='Number of\
                        negative samples with respect to positive samples (default: 0)')

    """
    Model Parameters
    """
    parser.add_argument('--vocab-size', type=int, default=1000000, help='Maximum vocabulary\
                        size (default: 1000000)')
    parser.add_argument('--embedding-len', type=int, default=512, help='Length of\
                        embeddings in model (default: 512)')
    parser.add_argument('--num-dense', type=int, default=1, help='Number of\
                        hidden dense layers in model (default: 1)')
    parser.add_argument('--num-units', type=int, default=64, help='Number of\
                        units for hidden dense layers in model (default: 64)')
    parser.add_argument('--dropout-rate', type=float, default=0.0, help='Dropout\
                        rate after each hidden layer (default: 0.0)')

    """
    Training Hyperparameters
    """
    parser.add_argument('-m', '--mode', default='cbow', choices=["cbow","skipgram"],
                        help='Training model node (default: cbow)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train for - iterations over the dataset (default: 10)')
    parser.add_argument('--batch-size', type=int, default=128,
                        metavar='N', help='number of examples in a training batch (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--optimizer', type=str, default="adam", metavar='O',
                        help='learning rate (default: adam)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    model_name = f"{args.mode}_{args.dataset_name}"
    

    # Load dataset
    print("Loading dataset...")
    corpus = load_dataset(f"{DATASETS_PATH}/{args.dataset_name}.txt")
    #print(dataset)

    # Load target words
    target_words = load_dataset(f"{TARGET_WORDS_PATH}/target_words_{args.dataset_name}.txt").split()

    # Set up a logs directory for Tensorboard
    log_dir='logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    print("Preprocessing dataset...")
    
    # Standardize text
    target_words = [standardization(word) for word in target_words]
    corpus = standardization(corpus)
    #print(corpus)

    # Select trainer
    if(args.mode == "cbow"):
        trainer_name = TWPModel
    elif(args.mode == "skipgram"):
        trainer_name = CPModel
    else:
        raise Exception("Invalid mode")
    
    # Create trainer
    trainer = trainer_name(vars(args))

    # Compute vocabulary
    print("Preparing text vectorizer...")
    trainer.compute_vocabulary(corpus)

    # Prepare model
    trainer.build_model()

    # Get vocabulary and word_index
    vocabulary = trainer.get_vocabulary()
    word_index = trainer.get_word_index()

    # Preprocess data
    X, y = trainer.preprocess_data(corpus)

    # Print samples
    print("Data samples:")
    print(f"{X[:2]=}")
    print(f"{y[:2]=}")
    #print("Vocabulary:",vocabulary)
    
    # Extract the dictionary which maps the word IDs to their embeddings
    embeddings = trainer.get_embedding_weights()
    #print("Embeddings:",embeddings)

    # Visualize TSNE
    if not os.path.exists(f"{PLOTS_PATH}/embeddings"):
        os.makedirs(f"{PLOTS_PATH}/embeddings")
    visualize_tsne_embeddings(target_words, embeddings, word_index, filename=f"{PLOTS_PATH}/embeddings/{args.mode}_{args.dataset_name}_before.png")

    # Train with a specific dataset split
    if args.split == "holdOut":
        train_ds, val_ds = create_training_sets(X,y)
        
        # Begin Training!
        history = trainer.train(train_ds,val_ds)

        # Plot training
        plot_training(history, model_name)

    else:
        # Begin Training!
        history = trainer.train((X,y),None)

    # Save model as a pickle file
    print("Saving model as a pickle file...")
    if not os.path.exists(MODEL_OUTPUT_PATH):
        os.makedirs(MODEL_OUTPUT_PATH)

    with open(f"{MODEL_OUTPUT_PATH}/{model_name}.pickle", "wb") as data_file:
        pickle.dump(trainer,data_file)

    # Extract the dictionary which maps the word IDs to their embeddings
    embeddings = trainer.get_embedding_weights()

    # Visualize TSNE
    visualize_tsne_embeddings(target_words, embeddings, word_index, filename=f"{PLOTS_PATH}/embeddings/{args.mode}_{args.dataset_name}_after.png")

    #####################
    
    # DUMP TO TENSORBOARD
    
    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for subwords in vocabulary:
            f.write("{}\n".format(subwords))

    # Save the weights we want to analyze as a variable. Note that the first
    # value represents any unknown word, which is not in the metadata, here
    # we will remove this value.
    print("Saving Data for TensorBoard...")
    weights = tf.Variable(trainer.get_embedding_weights()[1:])
    # Create a checkpoint from embedding, the filename and key are the
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    # Set up config.
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)