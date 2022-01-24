import datetime
import os
import random

import neptune.new as neptune
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import np_utils
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from protlearn.features import aaindex1, paac
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score, average_precision_score

from features_extraction import CT_processing, DPC_processing
from utils import create_dataset, categorical_probas_to_classes, calculate_performace

# Model architecture

input_1 = Input(shape=(1319)) 
dense11 = Dense(100, activation="relu")(input_1)
dense12 = (Dense(50, activation='relu'))(dense11)
drop2 = Dropout(0.1)(dense12)
output = Dense(17, activation="softmax")(drop2)
model = Model(inputs=[input_1], outputs=output)
# summarize layers
print(model.summary())


if __name__ == "__main__":

    path = "Quatpred/"
    if not os.path.exists(path):
        os.makedirs(path)

    Quat = open(path + "Quatpred_performance.txt" , "w")

    Quat.writelines(
        "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        + "\n"
    )
    Quat.write(
        "            sensitivity,     specificity,    precision,     acc,       mcc,       "
        + "\n"
    )
    Quat.writelines(
        "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        + "\n"
    )


    # init neptune logger
    run = neptune.init(
         project="sophiedalentour/QuaPred", 
         tags=["DPC", "PseAAC", "CT", "aaind", "muti_class_ classification"],
        )

    # set the seed
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    
    # training parameters
    BATCH_SIZE = 128
    NUM_EPOCHS = 10
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
        optimizer="adam",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )

    # create  dataset
    sequences, labels = create_dataset(data_path=DATA)
    
    # DPC 
    dip = DPC_processing(sequences)
    dipeptide= np.array(dip)
    print(dipeptide.shape) #(, 400)
	
    
    # PseAAC 
    seqs = sequences
    paac_comp, desc = paac(seqs, lambda_=3, remove_zero_cols=True)
    print(paac_comp.shape) #(, 23)
    
    # CT 
    CT = CT_processing(sequences)
    conjoint_triad= np.array(CT)
    print(conjoint_triad.shape) #(, 343)


    # AAindex 
    seqs = sequences
    aaind, inds = aaindex1(seqs, standardize='zscore')
    print(aaind.shape) #(, 553)
    
    # Dimension concatenation
    concat= np.concatenate(
        (
        
        dipeptide,
        paac_comp,
        conjoint_triad,
        aaind
        ), axis= 1
    )
    print(concat.shape) #(, 423)
    

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_labels = encoder.transform(labels)
    # convert integers to dummy variables (i.e. one hot encoded)
    one_hot_encoded = np_utils.to_categorical(encoded_labels)
    print(one_hot_encoded) #(,17)

    
    
    k_fold = 10
    scores = []

    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=1024)
    for ((train, test), k) in zip(skf.split(concat, encoded_labels), range(k_fold)):
        X_train = concat[train]
        print(X_train.shape)
        X_test = concat[test]
        print(X_test.shape)
        y_tr = encoded_labels[train]
        y_train= np_utils.to_categorical(y_tr)
        y_te = encoded_labels[test]
        y_test= np_utils.to_categorical(y_te)
        print(y_test)
        
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
    #run.stop()
     # prediction probability
        predictions = model.predict(X_test)
        y_class = categorical_probas_to_classes(predictions)

        # true_y_C_C=utils.categorical_probas_to_classes(true_y_C)
        true_y = categorical_probas_to_classes(y_test)
        (   
            sensitivity, 
            specificity,
            precision,
            acc,
            mcc,
        ) = calculate_performace(len(y_class), y_class, true_y)
        print("======================")
        print("======================")
        print(
            "Iter " + ", " + str(k + 1) + " of " + str(k_fold) + "cv:"
        )
        print(
            "\tsn='%0.4f', sp='%0.4f', precision='%0.4f',  acc='%0.4f', mcc='%0.4f'"
            % (sensitivity, specificity, precision, acc, mcc)
        )
        
        Quat.write(
            
            str(sensitivity) 
            + ","
            + str(specificity)
            + ","
            + str(precision)
            + ","
            + str(acc)
            + ","
            + str(mcc)
            + "\n"
        )
        scores.append(
            [sensitivity, specificity, precision,  acc, mcc]
        )
    scores = np.array(scores)

    print(len(scores))
    
    print(
        "sensitivity=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[0] * 100, np.std(scores, axis=0)[0] * 100)
    )
    print(
        "specificity=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[1] * 100, np.std(scores, axis=0)[1] * 100)
    )
    print(
        "precision=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[2] * 100, np.std(scores, axis=0)[2] * 100)
    )

    print(
        "acc=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[3] * 100, np.std(scores, axis=0)[3] * 100)
    )
    print(
        "mcc=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[4] * 100, np.std(scores, axis=0)[4] * 100)
    )
    


    Quat.write(
        "sensitivity=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[0] * 100, np.std(scores, axis=0)[0] * 100)
        + "\n"
    )
    Quat.write(
        "specificity=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[1] * 100, np.std(scores, axis=0)[1] * 100)
        + "\n"
    )
    Quat.write(
        "precision=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[2] * 100, np.std(scores, axis=0)[2] * 100)
        + "\n"
    )
    Quat.write(
        "acc=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[3] * 100, np.std(scores, axis=0)[3] * 100)
        + "\n"
    )
    Quat.write(
        "mcc=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[4] * 100, np.std(scores, axis=0)[4] * 100)
        + "\n"
    )

    Quat.close()
    

    run.stop()

