""" 
QuaPred training script using Bi-LSTM and esm1_t6_43M_UR50S
"""
import datetime
import os
import random


import neptune.new as neptune
import pandas as pd
import numpy as np
import tensorflow as tf
from biotransformers import BioTransformers
from keras.utils import np_utils
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, concatenate
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import AUC, BinaryAccuracy
from tensorflow.keras.models import Sequential
from protlearn.features import aaindex1, paac
from sklearn.model_selection import StratifiedKFold
from utils import create_dataset
from features_extraction import CT_processing, DPC_processing



# Model architecture
# first input model
def build_model(embedding_size):
    """QuaPred model
    dense layers model, to predict quaternary structure
    """
    model = Sequential(name="QuaPred_model")
    model.add(Dense(100, input_dim=embedding_size, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(17, activation='softmax'))
    return model


if __name__ == "__main__":

    # init neptune logger
    run = neptune.init(
         project="sophiedalentour/QuaPred", 
         tags=["bio-transformers", "muti_class_ classification"],
        )

    # set the seed
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # embedding and convolution parameters
    EMBEDDING_SIZE= 1319
    BIOTF_MODEL = "esm1_t6_43M_UR50S"
    BIOTF_POOLMODE = "cls"
    BIOTF_BS = 2

    # training parameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 200
    SAVED_MODEL_PATH = (
        "logs/model_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".hdf5"
    )
    DATA = "Uniprot_data/uniprot_dataset.csv"
    
  
    # save parameters in neptune
    run["hyper-parameters"] = {
        "encoding_mode": "bio-transformers",
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "embedding_size": EMBEDDING_SIZE,
        "num_epochs": NUM_EPOCHS,
        "saved_model_path": SAVED_MODEL_PATH,
        "data": DATA,
    }

    # build model
    model = build_model(EMBEDDING_SIZE)
    print(model.summary())

    # create  dataset
    sequences, labels = create_dataset(data_path=DATA)
    

    
    # sequences embeddings with biotransformers (input_1)
    """
    bio_trans = BioTransformers(backend=BIOTF_MODEL)

    sequences_embeddings = []

    for i in range(0, len(sequences), BIOTF_BS):
        batched_sequence = sequences[i:i + BIOTF_BS]
        batched_embedding = bio_trans.compute_embeddings(
            batched_sequence, pool_mode=(BIOTF_POOLMODE,), batch_size=BIOTF_BS
        )[
        BIOTF_POOLMODE
        ]  # (BIOTF_BS, 768)
        sequences_embeddings.append(batched_embedding)

    sequences_embeddings = tf.concat(sequences_embeddings, axis=0) # (7749, 768)
    sequences_embeddings= sequences_embeddings.numpy() #from tensor to numpy
    """
    
    # DPC (input_2)
    dip = DPC_processing(sequences)
    dipeptide= np.array(dip)
    print(dipeptide.shape) #(10237, 400)
	
    # CT (input_3)
    CT = CT_processing(sequences)
    conjoint_triad= np.array(CT)
    print(conjoint_triad.shape) #(10237, 343)


    # PseAAC (input_4)
    seqs = sequences
    paac_comp, desc = paac(seqs, lambda_=3, remove_zero_cols=True)
    print(paac_comp.shape) #(10237, 23)
    
    # AAindex (input_5)
    seqs = sequences
    aaind, inds = aaindex1(seqs, standardize='zscore')
    print(aaind.shape) #(10237, 553)

    
    # Dimension concatenation
    concat= np.concatenate(
        (
        dipeptide,
        paac_comp,
        aaind,
        conjoint_triad,
        ), axis= 1
    )
    print(concat)
    print(concat.shape) #(10237, 2087)


    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_labels = encoder.transform(labels)
    # convert integers to dummy variables (i.e. one hot encoded)
    one_hot_encoded = np_utils.to_categorical(encoded_labels)
    print(one_hot_encoded) #(10238, 17)

    """
    # split data 
    X_train, X_test, y_train, y_test = train_test_split(
            concat,
            one_hot_encoded, 
            test_size=0.2,
        )
    """
    # sequences reshape
    #x_train = X_train.reshape(
        #X_train.shape[0], 768, 1
    #)  
    #x_test = X_test.reshape(
        #X_test.shape[0], 768, 1
    #) 


    # compile the model
    cls = KerasClassifier(
        model=model, epochs=200, batch_size=BATCH_SIZE, verbose=0, 
        optimizer=keras.optimizers.SGD(learning_rate=0.001),
        loss=[BinaryCrossentropy, CategoricalCrossentropy],
        metrics=[BinaryAccuracy, AUC],
        callbacks= NeptuneCallback(run=run, base_namespace="metrics"),
        validation_split= 0.1
    )
    skf= StratifiedKFold(n_splits=2)
    for train, test in skf.split(concat,encoded_labels): 
        X_train = concat[train]
        print(X_train.shape)
        X_test = concat[test]
        print(X_test.shape)
        y_tr = encoded_labels[train]
        y_train= np_utils.to_categorical(y_tr)
        y_te = encoded_labels[test]
        y_test= np_utils.to_categorical(y_te)
        print(y_test)
        # fit the model 
        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)
        print(y_pred)
        print(y_pred.shape)
        y_true = y_test
        print(y_true.shape)
        print(X_train.shape)
        accuracy = accuracy_score(y_true, y_pred)
        print(accuracy)
        precision= precision_score(y_true, y_pred, average=None, zero_division=0)
        print(precision)
        label_names = ['monomer', 'homo-dimer', 'homo-trimer', 'homo-tetramer', 'homo-pentamer', 'homo-hexamer',
                        'homo-octamer', 'homo-decamer', 'homo-dodecamer', 
                        'hetero-dimer', 'hetero-trimer', 'hetero-tetramer', 'hetero-pentamer', 'hetero-hexamer',
                        'hetero-octamer', 'hetero-decamer', 'hetero-dedocamer']

        print(classification_report(y_true, y_pred,target_names=label_names, zero_division=0))
        
        # save model
        tf.keras.models.save_model(
            model,
            SAVED_MODEL_PATH,
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None
        )
    run.stop()
