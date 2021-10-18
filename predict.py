import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import glob
import time
import sys

import tqdm
import numpy as np
from data import *
from collections import Counter
from copy import deepcopy
from nltk.tokenize import word_tokenize
import re
from neural_networks.lmtc_networks.label_driven_classification import LabelDrivenClassification

from document_model import Document
from json_loader import JSONLoader

from tensorflow import keras
from vectorizer import W2VVectorizer, ELMoVectorizer, BERTVectorizer
import json


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

if len(sys.argv) > 1:
    if sys.argv[1] == "test":
        custom = True
else:
    custom = False

print(f" ---------  Is this a test? {custom}")

with open("data/datasets/EURLEX57K/EURLEX57K.json") as f:
    eurlex = json.load(f)
    eurovoc = {}
    for key in eurlex:
        eurovoc[key] = eurlex[key]["label"]
    eurovoc_reversed = {value : key for (key, value) in eurovoc.items()}

def load_dataset_s(docs):
    filenames = [os.path.join("data/datasets/EURLEX57K/train",doc) for doc in docs]
    loader = JSONLoader()
    documents = []
    #for filename in tqdm.tqdm(filenames):
    for filename in filenames:
        documents.append(loader.read_file(filename))
            
    return documents

def create_dataset(docs):
    """
    This mimicks load_dataset_s() but does not rely on JSONLoader
    """
    documents = []
    for doc in docs:
        with open(doc) as f:
            text = f.read()
        documents.append(Document(text, [], [], doc))
    
    return documents
        

def process_dataset(documents):
    """
        Process dataset documents (samples)
    """
    samples = []
    for document in documents:
        samples.append(document.tokens)

    del documents
    return samples

def encode_dataset(sequences):
    samples = vectorizer.vectorize_inputs(sequences=sequences,
                    max_sequence_size=5000)
    del sequences
    return samples


vectorizer = W2VVectorizer()
export_dir = "data/models/EURLEX57K_FLAT_LABEL_WISE_ATTENTION_NETWORK_bckp"
start = time.time()
model = keras.models.load_model(export_dir)
print(f"Model loaded in {round(time.time()-start, 3)}s")

## This allows us to try on our own docs
if custom == True:
    docs = ["tests/vonderleyen-wiki.txt", "tests/greendeal.txt"]
    documents = create_dataset(docs)
else:
    docs = sorted(["31958D1127(01).json", "31958R0001.json", "31961R0007.json", "31962D0403(01).json", "31962D1217(02).json", "31962L2005.json", "31962L2645.json", "31962R0025.json", "31962R0026.json", "31962R0027.json", "31962R0049.json", "31962R0059.json", "31963D0009.json", "31963L0261.json", "31963L0262.json", "31963L0474.json", "31963L0607.json", "31963R0099.json", "31964D0301.json", "31964D0350.json", "31964D0390.json", "31964L0054.json", "31964L0222.json", "31964L0224.json", "31964L0225.json", "31964L0427.json", "31964L0429.json", "31964R0087.json", "31964R0099.json", "31965L0001.json", "31965L0066.json", "31965L0264.json", "31965L0469.json", "31965R0010.json", "31965R0019.json", "31965R0051.json", "31965R0054.json", "31965R0174.json", "31966D0399.json", "31966D0556.json", "31966D0740.json", "31966L0162.json", "31966L0601.json", "31966R0017(01).json", "31966R0121.json", "31966R0172.json", "31966R0183.json", "31966R0211.json", "31967L0043.json", "31967L0428.json", "31967L0531.json", "31967L0532.json", "31967L0653.json", "31967R0115.json", "31967R0163.json", "31967R0164.json", "31967R0190.json", "31967R0283.json", "31967R0423.json", "31967R0467.json", "31967R0470.json", "31967R0614.json", "31967R0724.json", "31967R0765.json", "31967R0767.json", "31968D0189.json", "31968D0359.json", "31968D0361.json", "31968D0406.json", "31968D0416.json", "31968L0192.json", "31968L0221.json", "31968L0297.json", "31968L0360.json", "31968L0367.json", "31968L0368.json", "31968L0369.json", "31968L0415.json", "31968L0419.json", "31968R0190.json", "31968R0192.json", "31968R0246.json", "31968R0260.json", "31968R0261.json", "31968R0316.json", "31968R0391.json", "31968R0421.json", "31968R0431.json", "31968R0447.json", "31968R0564.json", "31968R0565.json", "31968R0821.json", "31968R0845.json", "31968R0876.json", "31968R0885.json", "31968R0937.json", "31968R0985.json", "31968R0989.json", "31968R0998.json", "31968R1014.json", "31968R1073.json", "31968R1077.json", "31968R1098.json", "31968R1216.json", "31968R1389.json", "31968R1397.json", "31968R1431.json", "31968R1469.json", "31968R1697.json", "31968R1767.json", "31968R2146.json", "31969D0266.json", "31969D0414.json", "31969L0060.json", "31969L0062.json", "31969L0064.json", "31969L0077.json", "31969L0082.json", "31969L0169.json", "31969L0463.json", "31969L0464.json", "31969L0465.json", "31969L0466.json", "31969L0493.json", "31969R0019.json", "31969R0098.json", "31969R0210.json", "31969R0448.json", "31969R0449.json", "31969R0549.json", "31969R0729.json", "31969R0878.json", "31969R0880.json", "31969R0955.json", "31969R0972.json", "31969R0990.json", "31969R1064.json", "31969R1211.json", "31969R1273.json", "31969R1353.json", "31969R1465.json", "31969R1467.json", "31969R1630.json", "31969R1913.json", "31969R2049.json", "31969R2118.json", "31969R2264.json", "31969R2488.json", "31969R2497.json", "31969R2511.json", "31969R2518.json", "31969R2602.json", "31969R2603.json", "31969R2604.json", "31970D0244.json", "31970D0372.json", "31970L0157.json", "31970L0189.json", "31970L0221.json", "31970L0222.json", "31970L0311.json", "31970L0357.json", "31970L0358.json", "31970L0359.json", "31970L0373.json", "31970L0387.json", "31970L0388.json", "31970L0451.json", "31970L0522.json", "31970L0523.json", "31970R0059.json", "31970R0251.json", "31970R0270.json", "31970R0491.json", "31970R0537.json", "31970R0757.json", "31970R1019.json", "31970R1107.json", "31970R1108.json", "31970R1249.json", "31970R1251.json", "31970R1285.json", "31970R1381.json", "31970R1411.json", "31970R1467.json", "31970R1469.json", "31970R1471.json", "31970R1484.json", "31970R1491.json", "31970R1520.json", "31970R1523.json", "31970R1524.json", "31970R1525.json", "31970R1562.json", "31970R1618.json", "31970R1698.json", "31970R1726.json", "31970R1728.json", "31970R2047.json", "31970R2048.json"])
    docs = docs[:1]
    documents = load_dataset_s(docs)

print(" ---------  Documents loaded")
val_samples = process_dataset(documents)
print(" ---------  Documents processed")
val_samples = encode_dataset(val_samples)
print(" ---------  Documents vectorised")



if os.path.exists("label_terms_text.txt") == True:
    with open("label_terms_text.txt") as f:
        label_terms_text = [line.strip() for line in f]
else:
    #with open("label_terms_text.txt", "w") as f:
    #    [f.write(item+"\n") for item in label_terms_text]
    sys.exit("Run test_simon.py to get the labels")

start = time.time()
predictions = model.predict(val_samples)
print(f"Prediction time for {len(docs)} document(s): {round(time.time()-start, 3)}s")

threshold = 0.33
pred_targets = (predictions > threshold).astype('int32')
print(" ---------  Results:")
for _, sub in enumerate(pred_targets[:10]):
    ot = np.where(sub == True)
    label_terms_text_labels = [label_terms_text[id_] for id_ in ot[0]]
    eurovoc_ids = [eurovoc_reversed[label] for label in label_terms_text_labels]
    print(docs[_], label_terms_text_labels, eurovoc_ids)
    #print()
