import datetime
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Flatten,
    Convolution1D,
    MaxPooling1D,
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
)
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from utils import create_dataset
from biotransformers import BioTransformers
from protlearn.features import aaindex1, paac
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from features_extraction import CT_processing, DPC_processing
# Model architecture
# first input model
input_1 = Input(shape=(2087, 1))  # embedding_layer(None, 1344, 1)
conv11 = Convolution1D(
    64,
    kernel_size=16,
    activation="relu",
    kernel_initializer="random_uniform",
    name="convolution_1d_layer1",
)(input_1)
pool11 = MaxPooling1D(pool_size=4)(conv11)
conv12 = Convolution1D(
    32,
    kernel_size=16,
    activation="relu",
    kernel_initializer="random_uniform",
    name="convolution_1d_layer2",
)(pool11)
pool12 = MaxPooling1D(pool_size=4)(conv12)
lstm11 = LSTM(32, return_sequences=False, name="lstm1")(pool12)
drop12 = Dropout(0.1)(lstm11)
flat1 = Flatten()(pool12)
hidden1 = Dense(10, activation="relu")(flat1)
drop1 = Dropout(0.1)(hidden1)
output = Dense(17, activation="sigmoid")(drop1)
model = Model(inputs=input_1, outputs=output)
# summarize layersa
print(model.summary())


if __name__ == "__main__":

    # init neptune logger
    run = neptune.init(
         project="sophiedalentour/QuaPred", 
         tags=["bio-transformers", "DPC", "PseAAC", "aaindex", "CT" "muti_class_ classification"],
        )

    # set the seed
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # embedding and convolution parameters
    BIOTF_MODEL = "esm1_t6_43M_UR50S"
    BIOTF_POOLMODE = "cls"
    BIOTF_BS = 2

    # training parameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 1000
    SAVED_MODEL_PATH = (
        "logs/model_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".hdf5"
    )
    DATA = "Uniprot_data/uniprot_dataset.csv"
    
  
    # save parameters in neptune
    run["hyper-parameters"] = {
        "encoding_mode": "bio-transformers",
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "saved_model_path": SAVED_MODEL_PATH,
        "data": DATA,
    }

    # build model
     # compile model
    model.compile(
        loss="binary_crossentropy",
        optimizer="adagrad",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )

    # create  dataset
    sequences, labels = create_dataset(data_path=DATA)

    
    # sequences embeddings with biotransformers (input_1)
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

    sequences_embeddings = tf.concat(sequences_embeddings, axis=0) # (10237, 768)
    sequences_embeddings= sequences_embeddings.numpy() #from tensor to numpy


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
        sequences_embeddings,
        conjoint_triad,
        dipeptide,
        paac_comp,
        aaind
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


    # split data 
    X_train, X_test, y_train, y_test = train_test_split(
            concat,
            one_hot_encoded, 
            test_size=0.2,
        )
    print(X_train.shape)
    print(X_test.shape)
    #sequences reshape
    x_train = X_train.reshape(
        X_train.shape[0], 2087, 1
    )  
    x_test = X_test.reshape(
        X_test.shape[0], 2087, 1
    ) 


    # define callbacks
    my_callbacks = [
        # ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1),
        # EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=1),
        ModelCheckpoint(
            monitor="val_accuracy",
            mode="max",
            filepath=SAVED_MODEL_PATH,
            save_best_only=True,
        ),
        NeptuneCallback(run=run, base_namespace="metrics"),
    ]

    # fit the model
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1,
        validation_data= (X_test, y_test)
        ,
        callbacks=my_callbacks,
    )
    run.stop()
