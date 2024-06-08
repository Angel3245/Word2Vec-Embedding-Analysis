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
from src.embeddings import *
from src.semantic_similarity import find_closest_words, plot_similarity

MODEL_OUTPUT_PATH = "./models"
PLOTS_OUTPUT_PATH = "./plots"
TARGET_WORDS_PATH = "./target_words"

def get_args():
    parser = argparse.ArgumentParser(description="Perform a qualitative analysis of the results. For target words, compute Semantic Similarity \
                                     with respect to the rest of the words in the vocabulary (cosine similarity) and visualize embeddings using T-SNE")

    """
    Model selection
    """
    parser.add_argument('--dataset-name', type=str, default='game_of_thrones',
                        help='dataset name (default: game_of_thrones)')
    parser.add_argument('-m', '--mode', default='cbow', choices=["cbow","skipgram"],
                        help='Visualizer node (default: cbow)')


    return parser.parse_args()

if __name__ == '__main__':
    
    args = get_args()

    # Load target words
    target_words = load_dataset(f"{TARGET_WORDS_PATH}/target_words_{args.dataset_name}.txt").split()

    # Load model
    print("Loading pickle model...")
    with open(str(f"{MODEL_OUTPUT_PATH}/{args.mode}_{args.dataset_name}.pickle"), "rb") as data_file:
        trainer = pickle.load(data_file)
    #model = keras.models.load_model(str(f"{datafolder}/Model_output/{pos_model}"))

    # Get vocabulary and word_index
    vocabulary = trainer.get_vocabulary()
    word_index = trainer.get_word_index()

    # Extract the dictionary which maps the word IDs to their embeddings
    embeddings = trainer.get_embedding_weights()

    # Compute semantic similarity
    for target_word in target_words:
        print("Target word:",target_word, end=" ")
        if target_word in word_index:
            similarity_dict = find_closest_words(embeddings,vocabulary ,word_index[target_word], top_n=10)
            print("- Most similar words:",list(similarity_dict.keys()))
            #print("Similarity:",list(similarity_dict.values()))
        #plot_similarity(similarity_dict,vocabulary,word_index[target_word])

    # Visualize TSNE
    visualize_all_tsne_embeddings(embeddings, word_index, vocabulary, target_words, filename=f"{PLOTS_OUTPUT_PATH}/{args.mode}_{args.dataset_name}.png")