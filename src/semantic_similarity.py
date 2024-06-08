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

import tensorflow as tf
import matplotlib.pyplot as plt

# Calculate cosine similarity
def cosine_similarity(v1, v2):
    dot_product = tf.reduce_sum(tf.multiply(v1, v2), axis=-1)
    v1_norm = tf.norm(v1, axis=-1)
    v2_norm = tf.norm(v2, axis=-1)
    return dot_product / (v1_norm * v2_norm)

# Function to find closest words
def find_closest_words(embeddings, vocabulary, target_word_index, top_n=10):
    # Get target embedding
    target_word_embedding = embeddings[target_word_index]

    # Compute cosine similarity between the vocabulary and the target embeddings
    similarities = cosine_similarity(embeddings, target_word_embedding)

    # Get most similar ones
    top_similarities, top_indices = tf.math.top_k(similarities, k=top_n+1)  # Adding 1 to exclude the word itself

    # Return a similarity dictionary {word: similarity with respect to the target word}
    similarity_dict = {vocabulary[i]: similarity.numpy() for i, similarity in zip(top_indices.numpy(),top_similarities) if i != target_word_index}  # Exclude the word itself
    return similarity_dict

def plot_similarity(similar_words_dict,vocabulary,target_word_index):
    # Plotting
    plt.bar(similar_words_dict.keys(), similar_words_dict.values())
    plt.xlabel('Words')
    plt.ylabel('Cosine Similarity')
    plt.title(f'Top words similar to {vocabulary[target_word_index]}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()