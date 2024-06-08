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
import keras
import random
import numpy as np
import tqdm
import re, string

def standardization(corpus):
    # Lowercasing
    corpus = corpus.lower()

    # Remove Punctuation & Special Characters
    punctuation_pattern = r'[^\w\s]'
    corpus = re.sub(punctuation_pattern, '', corpus)

    # Remove extra spaces
    corpus = re.sub(r'\s+',' ',corpus)

    return corpus

def to_skipgram_token(corpus,
    vocabulary_size,
    context_window_size=2,
    num_neg_samples=0,
    seed=None):
    """
    IMPLEMENTATION WITH INTEGER AS INPUT (for models using Tokenizer)
    
    Generates skipgram word pairs.

    This function transforms a sequence of words indexes (list of integer)
    into tuples of words of the form:

    - (word, word in the same window), with label 1 (positive samples).
    - (word, random word from the vocabulary), with label 0 (negative samples).

    Args:
        sequence: A word sequence (sentence), encoded as a list
            of word indices (integers). If using a `sampling_table`,
            word indices are expected to match the rank
            of the words in a reference dataset (e.g. 10 would encode
            the 10-th most frequently occurring token).
            Note that index 0 is expected to be a non-word and will be skipped.
        vocabulary_size: Int, maximum possible word index + 1
        window_size: Int, size of sampling windows (technically half-window).
            The window of a word `w_i` will be
            `[i - window_size, i + window_size+1]`.
        negative_samples: Float >= 0. 0 for no negative (i.e. random) samples.
            1 for same number as positive samples.
        seed: Random seed.

    Returns:
        couples, labels: where `couples` are int pairs and
            `labels` are either 0 or 1.
    """
    # Elements of each training example are appended to these lists.
    targets, contexts = [], []

    # Build the sampling table for `vocab_size` tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocabulary_size)

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, labels = tf.keras.preprocessing.sequence.skipgrams(
        corpus,
        vocabulary_size=vocabulary_size,
        sampling_table=sampling_table,
        window_size=context_window_size,
        negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    if num_neg_samples > 0:
        labels = []
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_neg_samples,
                unique=True,
                range_max=vocabulary_size,
                seed=seed,
                name="negative_sampling")

            # Build context and label vectors (for one target word)
            context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_neg_samples, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

        targets = np.array(targets)
        contexts = np.array(contexts)
        labels = np.array(labels)

    else:
        positive_skip_grams = np.array(positive_skip_grams)
        labels = np.array(labels)
        targets = positive_skip_grams[:,0]
        contexts = positive_skip_grams[:,1:]
    
    print('\n')
    print(f"targets.shape: {targets.shape}")
    print(f"contexts.shape: {contexts.shape}")
    print(f"labels.shape: {labels.shape}")

    return (targets, contexts), labels

def generate_negative_samples(positive_pairs, num_neg_samples, vocabulary):
    # Extract unique words
    words = vocabulary.copy()
    
    # Shuffle the words to create randomness
    random.shuffle(words)
    
    labels = []
    for pair in positive_pairs:
        words = vocabulary.copy()

        # Get positive context word
        pos_word = pair[1]

        words.remove(pos_word)
        
        pair += random.sample(words, num_neg_samples)

        # Shuffle the pair
        copy = pair[1:]
        random.shuffle(copy)
        pair[1:] = copy
        
        # Create label (1 positive + n negative)
        label = [0] * (num_neg_samples + 1) # Initialize zeros
        label[pair.index(pos_word)-1] = 1 # Set to 1 the positive word

        labels.append(label)
        
    return positive_pairs, labels

def to_skipgram_string(corpus,
    vocabulary,
    context_window_size=2,
    num_neg_samples=0,
    seed=None):
    """
    IMPLEMENTATION WITH STRING AS INPUT (for models using TextVectorization)
    
    Generates skipgram word pairs.

    This function transforms a sequence of words (list of strings)
    into tuples of words of the form:

    - (word, word in the same window), with label 1 (positive samples).
    - (word, random word from the vocabulary), with label 0 (negative samples).

    Args:
        sequence: A word sequence (sentence), encoded as a list
            of word indices (integers). If using a `sampling_table`,
            word indices are expected to match the rank
            of the words in a reference dataset (e.g. 10 would encode
            the 10-th most frequently occurring token).
            Note that index 0 is expected to be a non-word and will be skipped.
        vocabulary_size: Int, maximum possible word index + 1
        window_size: Int, size of sampling windows (technically half-window).
            The window of a word `w_i` will be
            `[i - window_size, i + window_size+1]`.
        negative_samples: Float >= 0. 0 for no negative (i.e. random) samples.
            1 for same number as positive samples.
        seed: Random seed.

    Returns:
        couples, labels: where `couples` are int pairs and
            `labels` are either 0 or 1.
    """
    # Generate skip-gram pairs from the corpus with a given window_size
    tokens = corpus.split()
    context = []
    for i, target_word in enumerate(tokens):
        # Iterate over words within the window around the target word
        for j in range(max(0, i - context_window_size), min(len(tokens), i + context_window_size + 1)):
            # Skip the target word itself
            if j != i:
                # Generate skipgrams
                skipgram = [target_word, tokens[j]]
                context.append(skipgram)

    if num_neg_samples > 0:
        # Generate negative samples
        context, labels = generate_negative_samples(context, num_neg_samples, vocabulary)
    else:
        # Generate labels when no negative samples are used
        labels = [1]*len(context)

    # Convert array of arrays to an array of strings
    context = [" ".join(words) for words in context]
    
    return np.array(context), np.array(labels)

def to_cbow(corpus,context_window_size=2):
    """Generate training pairs according to CBOW model"""
    # Divide corpus in words
    corpus = corpus.split()

    X, y = [], []
    for i in tqdm.tqdm(range(context_window_size, len(corpus) - context_window_size)):
        context = corpus[i - context_window_size : i] + corpus[i + 1 : i + context_window_size + 1]
        target = corpus[i]
        X.append(context)
        y.append(target)

    # Convert array of arrays to an array of strings
    X = [" ".join(words) for words in X]

    return np.array(X), np.array(y)

def create_training_sets(X,y, train_fraction=0.85):
    # Split the NumPy array into training and test sets
    # Determine the sizes of training, validation, and test sets
    num_samples = len(X)
    train_size = int(train_fraction * num_samples)

    # Generate random indices without replacement
    indices = np.random.permutation(num_samples)

    # Split indices for training, validation, and test sets
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Split input and target data using indices
    return (X[train_indices],y[train_indices]),(X[val_indices],y[val_indices])
