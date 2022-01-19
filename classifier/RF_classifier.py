import datetime
import os
import random
import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf
from sklearn.preprocessing import scale,StandardScaler 
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from utils import create_dataset
from biotransformers import BioTransformers
from protlearn.features import aaindex1, paac
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from features_extraction import CT_processing, DPC_processing
from sklearn import metrics


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
        "num_epochs": NUM_EPOCHS,
        "saved_model_path": SAVED_MODEL_PATH,
        "data": DATA,
    }

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

    sequences_embeddings = tf.concat(sequences_embeddings, axis=0) # (7749, 768)
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


# split data 
    X_train, X_test, y_train, y_test = train_test_split(
            concat,
            one_hot_encoded, 
            test_size=0.2,
        )

cv_clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=11, 
                                min_samples_split=2, min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, max_features='sqrt', 
                                max_leaf_nodes=None,  bootstrap=True, 
                                oob_score=False, n_jobs=1, random_state=None, verbose=0, 
                                warm_start=False, class_weight=None)



hist=cv_clf.fit(X_train, y_train)
y_pred=cv_clf.predict(X_test)
print(y_pred)
    
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))