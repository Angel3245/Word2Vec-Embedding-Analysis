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
import keras.backend as K
import numpy as np
from keras import Input
from keras.metrics import CosineSimilarity
from keras.layers import Dense, Lambda, Embedding, Dropout, TextVectorization, Reshape, Dot, Flatten, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import KFold
from .preprocess import to_cbow, to_skipgram_token, to_skipgram_string

class TWPModel(object):
    """
    Class for CBOW model tokenization (using TextVectorization) and training.
    """
    def __init__(self, args):
        """
        Initialize the CBOW trainer object.

        Args:
            args (dict): Dictionary containing model configuration parameters.
        """
        self.model = None
        self.args = args

    def compute_vocabulary(self,corpus):
        """
        Compute the vocabulary from the provided corpus.

        Args:
            corpus (str): Text corpus for vocabulary computation.

        Returns:
            TextVectorization: TextVectorization object with the computed vocabulary.
        """
        self.text_vectorizer = TextVectorization(max_tokens=self.args["vocab_size"], output_sequence_length=self.args["window_size"]*2, output_mode='int', name="text_vectorizer")
        self.text_vectorizer.adapt([corpus])

        print("Text vectorizer layer prepared.\n")
        print("Text_vectorizer vocabulary_size:",self.text_vectorizer.vocabulary_size())

        return self.text_vectorizer
    
    def build_model(self):
        """
        Build the TWP (CBOW) model based on the configured parameters.
        """
        # Create an Input layer
        inputs = Input(shape=(1,), dtype=tf.string)

        # A TextVectorizer layer
        x = self.text_vectorizer(inputs)

        # An Embedding layer
        x = Embedding(self.text_vectorizer.vocabulary_size(), self.args["embedding_len"],
                      mask_zero=True, name="embedding")(x)

        # A Lambda layer
        x = Lambda(lambda x: K.mean(x, axis=1), output_shape=None, mask=None, arguments=None)(x)

        # Add Dense layers 
        for _ in range(self.args["num_dense"]):
            x = Dense(units=self.args["num_units"], activation='relu')(x)
            # Add Dropout after each Dense layer
            x = Dropout(self.args["dropout_rate"])(x)  

        # Output layer
        output = Dense(units=self.text_vectorizer.vocabulary_size(), activation='softmax', name="output")(x)

        # Create the model
        self.model = tf.keras.Model(inputs=inputs, outputs=output)

        print(self.model.summary())
        keras.utils.plot_model(self.model, show_shapes=True)

        # Release TextVectorizer memory
        self.text_vectorizer = None

    
    def train(self, train_sets, val_sets):
        """
        Train the model.

        Args:
            train_sets (tuple): Tuple containing training input sets.
            val_sets (tuple): Tuple containing validation input sets (optional).

        Returns:
            History: Training history.
        """
        # Load a TextVectorizer for converting targets to integer representations using the vocabulary from the model's TextVectorizer
        vocabulary = self.model.get_layer("text_vectorizer").get_vocabulary()
        text_vectorizer = TextVectorization(max_tokens=self.args["vocab_size"], output_sequence_length=1, output_mode='int', vocabulary=vocabulary)
        
        # Prepare training input in batches
        train_ds = tf.data.Dataset.from_tensor_slices((train_sets[0],text_vectorizer(train_sets[1])))
        train_ds = train_ds.batch(self.args["batch_size"])

        #val_ds = tf.data.Dataset.from_tensor_slices((val_sets[0],val_sets[1]))
        #val_ds = val_ds.batch(hyperparameters["batch_size"])

        # Compile the model
        self.model.compile(loss='sparse_categorical_crossentropy',
              optimizer=self.args["optimizer"],
              metrics=["accuracy"])

        # simple early stopping
        #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

        # Train the model and show loss and accuracy at the end of each epoch
        if val_sets is None:
            history = self.model.fit(train_ds,
                epochs=self.args["epochs"]
            )
        else:
            history = self.model.fit(train_ds,
                epochs=self.args["epochs"], validation_data=(val_sets[0],text_vectorizer(val_sets[1]))
                #callbacks=[es]
            )

        return history
    
    def cross_val_train(self, dataset, num_folds=10):
        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=num_folds, shuffle=True)

        # Define per-fold score containers
        acc_per_fold = []
        loss_per_fold = []

        # K-fold Cross Validation model evaluation
        fold_no = 1
        for train, val in kfold.split(dataset[0], dataset[1]):
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')
            
            history = self.train((dataset[0][train],dataset[1][train]),(dataset[0][val],dataset[1][val]))
            
            # Get generalization metrics
            acc_per_fold.append(min(history.history['val_accuracy']))
            loss_per_fold.append(min(history.history['val_loss']))

            # Increase fold number
            fold_no = fold_no + 1

            # Clear the previous weights
            keras.backend.clear_session()

        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')

        return np.mean(loss_per_fold), np.mean(acc_per_fold)

        
    def evaluate(self, test_sets):
        """
        Evaluate the model.

        Args:
            test_sets (tuple): Tuple containing test input sets.

        Returns:
            list: Evaluation metrics.
        """
        # Load a TextVectorizer for converting targets to integer representations using the vocabulary from the model's TextVectorizer
        vocabulary = self.model.get_layer("text_vectorizer").get_vocabulary()
        text_vectorizer = TextVectorization(max_tokens=self.args["vocab_size"], output_sequence_length=1, output_mode='int', vocabulary=vocabulary)
        
        # Prepare test input in batches
        ds = tf.data.Dataset.from_tensor_slices((test_sets[0],text_vectorizer(test_sets[1])))
        ds = ds.batch(self.args["batch_size"])
        
        return self.model.evaluate(ds)

    def predict(self, input):
        """
        Make predictions using the trained model.

        Args:
            input: Input data for prediction.

        Returns:
            numpy.ndarray: Model predictions.
        """
        vocabulary = self.model.get_layer("text_vectorizer").get_vocabulary()
        predictions = self.model.predict(input)

        return [vocabulary[prediction] for prediction in predictions.argmax(axis=1)]

    def get_vocabulary(self):
        """
        Get the vocabulary from the TextVectorization layer.

        Returns:
            list: Vocabulary.
        """
        return self.model.get_layer("text_vectorizer").get_vocabulary()
    
    def get_embedding_weights(self):
        """
        Get the embedding weights.

        Returns:
            numpy.ndarray: Embedding weights.
        """
        return self.model.get_layer("embedding").get_weights()[0]
    
    def get_word_index(self):
        """
        Get the word index of the vocabulary.

        Returns:
            dict: Word index.
        """
        return {word: index for index, word in enumerate(self.get_vocabulary())}

    def preprocess_data(self, corpus):
        """
        Preprocess the data to cbow format.

        Args:
            corpus (str): Text corpus.

        Returns:
            tuple: Preprocessed data.
        """
        return to_cbow(corpus, self.args["window_size"])

class CPModel(object):

    """
    Class for Skipgram model tokenization (using TextVectorization) and training.
    """
    def __init__(self, args):
        """
        Initialize the skipgram trainer object.

        Args:
            args (dict): Dictionary containing model configuration parameters.
        """
        self.model = None
        self.args = args

    def compute_vocabulary(self,corpus):
        """
        Compute the vocabulary from the provided corpus.

        Args:
            corpus (str): Text corpus for vocabulary computation.

        Returns:
            TextVectorization: TextVectorization object with the computed vocabulary.
        """
        self.text_vectorizer = TextVectorization(max_tokens=self.args["vocab_size"], output_sequence_length=self.args["neg_samples"]+2, output_mode='int', name="text_vectorizer")
        self.text_vectorizer.adapt([corpus])

        print("Text vectorizer layer prepared.\n")
        print("Text_vectorizer vocabulary_size:",self.text_vectorizer.vocabulary_size())

        return self.text_vectorizer
    
    def build_model(self):
        """
        Build the CP (skipgram) model based on the configured parameters.
        """
        #################
        # PROCESS INPUT #
        #################
        # Create an Input layer
        inputs = Input(shape=(1,), dtype=tf.string)

        x = self.text_vectorizer(inputs)

        # An Embedding layer
        x = Embedding(self.text_vectorizer.vocabulary_size(), self.args["embedding_len"],
                      mask_zero=True, name="embedding")(x)
        
        # Split the input tensor by columns
        x = tf.split(x, num_or_size_splits=self.args["neg_samples"]+2, axis=1)

        ##################
        # PROCESS TARGET #
        ##################
        x1 = Reshape((-1, 1))(x[0])

        ###################
        # PROCESS CONTEXT #
        ###################
        if self.args["neg_samples"] > 0:
            # Concatenate all contexts
            x2 = Concatenate(axis=1)(x[1:])

        else:
            x2 = x[1]

        x2 = Reshape((-1, 1+self.args["neg_samples"]))(x2)

        x = Dot(axes=1)([x1, x2])

        # Flatten layer to transform output shape
        x = Flatten()(x)

        # Add Dense layers 
        for _ in range(self.args["num_dense"]):
            x = Dense(units=self.args["num_units"], activation='relu')(x)
            # Add Dropout after each Dense layer
            x = Dropout(self.args["dropout_rate"])(x)  

        # Output layer
        output = Dense(units=1+self.args["neg_samples"], activation='sigmoid', name="output")(x)

        # Create the model
        self.model = tf.keras.Model(inputs=inputs, outputs=output)

        print(self.model.summary())
        keras.utils.plot_model(self.model, show_shapes=True)

        # Release TextVectorization memory
        self.text_vectorizer = None

    
    def train(self, train_sets, val_sets):
        """
        Train the model.

        Args:
            train_sets (tuple): Tuple containing training input sets.
            val_sets (tuple): Tuple containing validation input sets (optional).

        Returns:
            History: Training history.
        """
        # Prepare training input in batches
        train_ds = tf.data.Dataset.from_tensor_slices((train_sets[0],train_sets[1]))
        train_ds = train_ds.batch(self.args["batch_size"])

        #val_ds = tf.data.Dataset.from_tensor_slices((val_sets[0],val_sets[1]))
        #val_ds = val_ds.batch(hyperparameters["batch_size"])

        # Compile the model
        self.model.compile(loss='binary_crossentropy',
              optimizer=self.args["optimizer"],
              metrics=["accuracy"])

        # simple early stopping
        #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

        # Train the model and show loss and accuracy at the end of each epoch
        if val_sets is None:
            history = self.model.fit(train_ds,
                epochs=self.args["epochs"]
            )
        else:
            history = self.model.fit(train_ds,
                epochs=self.args["epochs"], validation_data=(val_sets[0],val_sets[1])
                #callbacks=[es]
            )

        return history
        
    def evaluate(self, test_sets):
        """
        Evaluate the model.

        Args:
            test_sets (tuple): Tuple containing test input sets.

        Returns:
            list: Evaluation metrics.
        """
        # Load a TextVectorizer for converting targets to integer representations using the vocabulary from the model's TextVectorizer
        vocabulary = self.model.get_layer("text_vectorizer").get_vocabulary()
        text_vectorizer = TextVectorization(max_tokens=self.args["vocab_size"], output_sequence_length=None, output_mode='int', vocabulary=vocabulary)
        
        # Prepare test input in batches
        ds = tf.data.Dataset.from_tensor_slices((test_sets[0],text_vectorizer(test_sets[1])))
        ds = ds.batch(self.args["batch_size"])
        
        return self.model.evaluate(ds)

    def predict(self, input):
        """
        Make predictions using the trained model.

        Args:
            input: Input data for prediction.

        Returns:
            numpy.ndarray: Model predictions.
        """
        return self.model.predict(input)
    
    def cross_val_train(self, dataset, num_folds=10):
        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=num_folds, shuffle=True)

        # Define per-fold score containers
        acc_per_fold = []
        loss_per_fold = []

        # K-fold Cross Validation model evaluation
        fold_no = 1
        for train, val in kfold.split(dataset[0], dataset[1]):
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')
            
            history = self.train((dataset[0][train],dataset[1][train]),(dataset[0][val],dataset[1][val]))
            
            # Get generalization metrics
            acc_per_fold.append(min(history.history['val_accuracy']))
            loss_per_fold.append(min(history.history['val_loss']))

            # Increase fold number
            fold_no = fold_no + 1

            # Clear the previous weights
            keras.backend.clear_session()

        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')

        return np.mean(loss_per_fold), np.mean(acc_per_fold)
    
    def preprocess_data(self, corpus):
        """
        Preprocess the data to skipgram format.

        Args:
            corpus (str): Text corpus.

        Returns:
            tuple: Preprocessed data.
        """
        return to_skipgram_string(corpus, self.get_vocabulary(), self.args["window_size"], self.args["neg_samples"], seed=self.args["seed"])
    
    def get_vocabulary(self):
        """
        Get the vocabulary from the TextVectorization layer.

        Returns:
            list: Vocabulary.
        """
        return self.model.get_layer("text_vectorizer").get_vocabulary()
    
    def get_word_index(self):
        """
        Get the word index of the vocabulary.

        Returns:
            dict: Word index.
        """
        return {word: index for index, word in enumerate(self.get_vocabulary())}
    
    def get_embedding_weights(self):
        """
        Get the embedding weights.

        Returns:
            numpy.ndarray: Embedding weights.
        """
        return self.model.get_layer("embedding").get_weights()[0]
    
class CPModel_Tokenizer(object):

    """
    ALTERNATIVE IMPLEMENTATION USING Tokenizer (UNUSED)
    Class for Skipgram model tokenization and training.
    """
    def __init__(self, args):
        """
        Initialize the skipgram object.

        Args:
            args (dict): Dictionary containing model configuration parameters.
        """
        self.model = None
        self.args = args

    def compute_vocabulary(self,corpus):
        """
        Compute the vocabulary from the provided corpus.

        Args:
            corpus (str): Text corpus for vocabulary computation.

        Returns:
            Tokenizer: TextTokenizer object with the computed vocabulary.
        """
        self.text_vectorizer = Tokenizer()
        self.text_vectorizer.fit_on_texts([corpus])
        
        word_index = self.text_vectorizer.word_index
        self.vocab_size = len(word_index) + 1

        print("Text vectorizer layer prepared.\n")
        print("Text_vectorizer vocabulary_size:",self.vocab_size)

        return self.text_vectorizer
    
    def tokenize(self,corpus):
        """
        Tokenize the provided corpus.

        Args:
            corpus (str): Text corpus to tokenize.

        Returns:
            list: Tokenized sequences of the corpus.
        """
        return self.text_vectorizer.texts_to_sequences([corpus])[0]
    
    def build_model(self):
        """
        Build the CP (skipgram) model based on the configured parameters.
        """
        #################
        # PROCESS INPUT #
        #################
        # Create an Input layer
        inputs_target = Input(shape=(1,), dtype=tf.int64)
        inputs_context = Input(shape=(1+self.args["neg_samples"],), dtype=tf.int64)

        #################
        # DEFINE LAYERS #
        #################
        # An Embedding layer
        embedding = Embedding(self.vocab_size, self.args["embedding_len"],
                      mask_zero=True, name="embedding")

        ##################
        # PROCESS TARGET #
        ##################

        # An Embedding layer
        target = embedding(inputs_target)

        # A Reshape layer
        target = Reshape((-1,1))(target)

        ####################
        # PROCESS CONTEXTS #
        ####################

        # An Embedding layer
        context = embedding(inputs_context)

        # A Reshape layer
        context = Reshape((-1, 1+self.args["neg_samples"]))(context)

        # A Dot layer
        x = Dot(axes=[1, 1])([target, context])

        ###################
        # PROCESS RESULTS #
        ###################
        # Flatten layer to transform output shape
        x = Flatten()(x)

        # Add Dense layers 
        for _ in range(self.args["num_dense"]):
            x = Dense(units=self.args["num_units"], activation='relu')(x)
            # Add Dropout after each Dense layer
            x = Dropout(self.args["dropout_rate"])(x)  

        ##########
        # OUTPUT #
        ##########
        # Output layer
        output = Dense(units=1+self.args["neg_samples"], activation='sigmoid', name="output")(x)

        # Create the model
        self.model = tf.keras.Model(inputs=[inputs_target,inputs_context], outputs=output)

        print(self.model.summary())
        keras.utils.plot_model(self.model, show_shapes=True)

    
    def train(self, train_sets, val_sets):
        """
        Train the CP model.

        Args:
            train_sets (tuple): Tuple containing training input sets.
            val_sets (tuple): Tuple containing validation input sets (optional).

        Returns:
            History: Training history.
        """
        # Prepare training input in batches
        train_ds = tf.data.Dataset.from_tensor_slices((train_sets[0],train_sets[1]))
        train_ds = train_ds.batch(self.args["batch_size"])

        #val_ds = tf.data.Dataset.from_tensor_slices((val_sets[0],val_sets[1]))
        #val_ds = val_ds.batch(hyperparameters["batch_size"])

        # Compile the model
        self.model.compile(loss='binary_crossentropy',
              optimizer=self.args["optimizer"],
              metrics=["accuracy"])

        # simple early stopping
        #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

        # Train the model and show loss and accuracy at the end of each epoch
        if val_sets is None:
            history = self.model.fit(train_ds,
                epochs=self.args["epochs"]
            )
        else:
            history = self.model.fit(train_ds,
                epochs=self.args["epochs"], validation_data=(val_sets[0],val_sets[1])
                #callbacks=[es]
            )

        return history
        
    def evaluate(self, test_sets):
        """
        Evaluate the CP model.

        Args:
            test_sets (tuple): Tuple containing test input sets.

        Returns:
            list: Evaluation metrics.
        """
        # Load a TextVectorizer for converting targets to integer representations using the vocabulary from the model's TextVectorizer
        vocabulary = self.text_vectorizer.word_index
        text_vectorizer = TextVectorization(max_tokens=self.args["vocab_size"], output_sequence_length=None, output_mode='int', vocabulary=vocabulary)
        
        # Prepare test input in batches
        ds = tf.data.Dataset.from_tensor_slices((test_sets[0],text_vectorizer(test_sets[1])))
        ds = ds.batch(self.args["batch_size"])
        
        return self.model.evaluate(ds)

    def predict(self, input):
        """
        Make predictions using the trained CP model.

        Args:
            input: Input data for prediction.

        Returns:
            numpy.ndarray: Model predictions.
        """
        return self.model.predict(input)

    
    def get_vocabulary(self):
        """
        Get the vocabulary.

        Returns:
            list: Vocabulary.
        """
        return list(self.text_vectorizer.word_index.keys())
    
    def get_word_index(self):
        """
        Get the word index.

        Returns:
            dict: Word index.
        """
        return self.text_vectorizer.word_index
    
    def get_embedding_weights(self):
        """
        Get the embedding weights.

        Returns:
            numpy.ndarray: Embedding weights.
        """
        return self.model.get_layer("embedding").get_weights()[0]
    
    def preprocess_data(self, corpus):
        """
        Preprocess the data.

        Args:
            corpus (str): Text corpus.

        Returns:
            tuple: Preprocessed data.
        """
        corpus = self.tokenize(corpus)

        return to_skipgram_token(corpus, self.args["vocab_size"], self.args["window_size"], self.args["neg_samples"], seed=self.args["seed"])