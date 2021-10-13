import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import glob
from json_loader import JSONLoader
import tqdm
import numpy as np
from data import *
from collections import Counter
from copy import deepcopy



gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

from document_model import Document
from json_loader import JSONLoader

from tensorflow import keras
from vectorizer import W2VVectorizer, ELMoVectorizer, BERTVectorizer
import json

with open("data/datasets/EURLEX57K/EURLEX57K.json") as f:
    eurlex = json.load(f)

for i in list(eurlex.keys())[:10]:
    print(i, eurlex[i]["label"])

vectorizer = W2VVectorizer()
#vectorizer = ELMoVectorizer()
#vectorizer = BERTVectorizer()

export_dir = "data/models/EURLEX57K_FLAT_LABEL_WISE_ATTENTION_NETWORK_bckp"
model2 = keras.models.load_model(export_dir)

sentences = []
docs = sorted(["31958D1127(01).json", "31958R0001.json", "31961R0007.json", "31962D0403(01).json", "31962D1217(02).json", "31962L2005.json", "31962L2645.json", "31962R0025.json", "31962R0026.json", "31962R0027.json", "31962R0049.json", "31962R0059.json", "31963D0009.json", "31963L0261.json", "31963L0262.json", "31963L0474.json", "31963L0607.json", "31963R0099.json", "31964D0301.json", "31964D0350.json", "31964D0390.json", "31964L0054.json", "31964L0222.json", "31964L0224.json", "31964L0225.json", "31964L0427.json", "31964L0429.json", "31964R0087.json", "31964R0099.json", "31965L0001.json", "31965L0066.json", "31965L0264.json", "31965L0469.json", "31965R0010.json", "31965R0019.json", "31965R0051.json", "31965R0054.json", "31965R0174.json", "31966D0399.json", "31966D0556.json", "31966D0740.json", "31966L0162.json", "31966L0601.json", "31966R0017(01).json", "31966R0121.json", "31966R0172.json", "31966R0183.json", "31966R0211.json", "31967L0043.json", "31967L0428.json", "31967L0531.json", "31967L0532.json", "31967L0653.json", "31967R0115.json", "31967R0163.json", "31967R0164.json", "31967R0190.json", "31967R0283.json", "31967R0423.json", "31967R0467.json", "31967R0470.json", "31967R0614.json", "31967R0724.json", "31967R0765.json", "31967R0767.json", "31968D0189.json", "31968D0359.json", "31968D0361.json", "31968D0406.json", "31968D0416.json", "31968L0192.json", "31968L0221.json", "31968L0297.json", "31968L0360.json", "31968L0367.json", "31968L0368.json", "31968L0369.json", "31968L0415.json", "31968L0419.json", "31968R0190.json", "31968R0192.json", "31968R0246.json", "31968R0260.json", "31968R0261.json", "31968R0316.json", "31968R0391.json", "31968R0421.json", "31968R0431.json", "31968R0447.json", "31968R0564.json", "31968R0565.json", "31968R0821.json", "31968R0845.json", "31968R0876.json", "31968R0885.json", "31968R0937.json", "31968R0985.json", "31968R0989.json", "31968R0998.json", "31968R1014.json", "31968R1073.json", "31968R1077.json", "31968R1098.json", "31968R1216.json", "31968R1389.json", "31968R1397.json", "31968R1431.json", "31968R1469.json", "31968R1697.json", "31968R1767.json", "31968R2146.json", "31969D0266.json", "31969D0414.json", "31969L0060.json", "31969L0062.json", "31969L0064.json", "31969L0077.json", "31969L0082.json", "31969L0169.json", "31969L0463.json", "31969L0464.json", "31969L0465.json", "31969L0466.json", "31969L0493.json", "31969R0019.json", "31969R0098.json", "31969R0210.json", "31969R0448.json", "31969R0449.json", "31969R0549.json", "31969R0729.json", "31969R0878.json", "31969R0880.json", "31969R0955.json", "31969R0972.json", "31969R0990.json", "31969R1064.json", "31969R1211.json", "31969R1273.json", "31969R1353.json", "31969R1465.json", "31969R1467.json", "31969R1630.json", "31969R1913.json", "31969R2049.json", "31969R2118.json", "31969R2264.json", "31969R2488.json", "31969R2497.json", "31969R2511.json", "31969R2518.json", "31969R2602.json", "31969R2603.json", "31969R2604.json", "31970D0244.json", "31970D0372.json", "31970L0157.json", "31970L0189.json", "31970L0221.json", "31970L0222.json", "31970L0311.json", "31970L0357.json", "31970L0358.json", "31970L0359.json", "31970L0373.json", "31970L0387.json", "31970L0388.json", "31970L0451.json", "31970L0522.json", "31970L0523.json", "31970R0059.json", "31970R0251.json", "31970R0270.json", "31970R0491.json", "31970R0537.json", "31970R0757.json", "31970R1019.json", "31970R1107.json", "31970R1108.json", "31970R1249.json", "31970R1251.json", "31970R1285.json", "31970R1381.json", "31970R1411.json", "31970R1467.json", "31970R1469.json", "31970R1471.json", "31970R1484.json", "31970R1491.json", "31970R1520.json", "31970R1523.json", "31970R1524.json", "31970R1525.json", "31970R1562.json", "31970R1618.json", "31970R1698.json", "31970R1726.json", "31970R1728.json", "31970R2047.json", "31970R2048.json"])

def load_dataset_s(docs):
    #docs = ["31958D1127(01).json", "31958R0001.json", "31961R0007.json", "31962D0403(01).json", "31962D1217(02).json", "31962L2005.json", "31962L2645.json", "31962R0025.json", "31962R0026.json", "31962R0027.json", "31962R0049.json", "31962R0059.json", "31963D0009.json", "31963L0261.json", "31963L0262.json", "31963L0474.json", "31963L0607.json", "31963R0099.json", "31964D0301.json", "31964D0350.json", "31964D0390.json", "31964L0054.json", "31964L0222.json", "31964L0224.json", "31964L0225.json", "31964L0427.json", "31964L0429.json", "31964R0087.json", "31964R0099.json", "31965L0001.json", "31965L0066.json", "31965L0264.json", "31965L0469.json", "31965R0010.json", "31965R0019.json", "31965R0051.json", "31965R0054.json", "31965R0174.json", "31966D0399.json", "31966D0556.json", "31966D0740.json", "31966L0162.json", "31966L0601.json", "31966R0017(01).json", "31966R0121.json", "31966R0172.json", "31966R0183.json", "31966R0211.json", "31967L0043.json", "31967L0428.json", "31967L0531.json", "31967L0532.json", "31967L0653.json", "31967R0115.json", "31967R0163.json", "31967R0164.json", "31967R0190.json", "31967R0283.json", "31967R0423.json", "31967R0467.json", "31967R0470.json", "31967R0614.json", "31967R0724.json", "31967R0765.json", "31967R0767.json", "31968D0189.json", "31968D0359.json", "31968D0361.json", "31968D0406.json", "31968D0416.json", "31968L0192.json", "31968L0221.json", "31968L0297.json", "31968L0360.json", "31968L0367.json", "31968L0368.json", "31968L0369.json", "31968L0415.json", "31968L0419.json", "31968R0190.json", "31968R0192.json", "31968R0246.json", "31968R0260.json", "31968R0261.json", "31968R0316.json", "31968R0391.json", "31968R0421.json", "31968R0431.json", "31968R0447.json", "31968R0564.json", "31968R0565.json", "31968R0821.json", "31968R0845.json", "31968R0876.json", "31968R0885.json", "31968R0937.json", "31968R0985.json", "31968R0989.json", "31968R0998.json", "31968R1014.json", "31968R1073.json", "31968R1077.json", "31968R1098.json", "31968R1216.json", "31968R1389.json", "31968R1397.json", "31968R1431.json", "31968R1469.json", "31968R1697.json", "31968R1767.json", "31968R2146.json", "31969D0266.json", "31969D0414.json", "31969L0060.json", "31969L0062.json", "31969L0064.json", "31969L0077.json", "31969L0082.json", "31969L0169.json", "31969L0463.json", "31969L0464.json", "31969L0465.json", "31969L0466.json", "31969L0493.json", "31969R0019.json", "31969R0098.json", "31969R0210.json", "31969R0448.json", "31969R0449.json", "31969R0549.json", "31969R0729.json", "31969R0878.json", "31969R0880.json", "31969R0955.json", "31969R0972.json", "31969R0990.json", "31969R1064.json", "31969R1211.json", "31969R1273.json", "31969R1353.json", "31969R1465.json", "31969R1467.json", "31969R1630.json", "31969R1913.json", "31969R2049.json", "31969R2118.json", "31969R2264.json", "31969R2488.json", "31969R2497.json", "31969R2511.json", "31969R2518.json", "31969R2602.json", "31969R2603.json", "31969R2604.json", "31970D0244.json", "31970D0372.json", "31970L0157.json", "31970L0189.json", "31970L0221.json", "31970L0222.json", "31970L0311.json", "31970L0357.json", "31970L0358.json", "31970L0359.json", "31970L0373.json", "31970L0387.json", "31970L0388.json", "31970L0451.json", "31970L0522.json", "31970L0523.json", "31970R0059.json", "31970R0251.json", "31970R0270.json", "31970R0491.json", "31970R0537.json", "31970R0757.json", "31970R1019.json", "31970R1107.json", "31970R1108.json", "31970R1249.json", "31970R1251.json", "31970R1285.json", "31970R1381.json", "31970R1411.json", "31970R1467.json", "31970R1469.json", "31970R1471.json", "31970R1484.json", "31970R1491.json", "31970R1520.json", "31970R1523.json", "31970R1524.json", "31970R1525.json", "31970R1562.json", "31970R1618.json", "31970R1698.json", "31970R1726.json", "31970R1728.json", "31970R2047.json", "31970R2048.json"]
    docs = [os.path.join("data/datasets/EURLEX57K/train",doc) for doc in docs]
    loader = JSONLoader()
    documents = []
    for filename in tqdm.tqdm(sorted(docs[:20])):
        documents.append(loader.read_file(filename))
            
    return documents

def process_dataset(documents):
    """
        Process dataset documents (samples) and create targets
        :param documents: list of Document objects
        :return: samples, targets
        """
    samples = []
    targets = []
    for document in documents:
        samples.append(document.tokens)
        targets.append(document.tags)

    del documents
    return samples, targets

def encode_dataset(sequences, tags):
    samples = vectorizer.vectorize_inputs(sequences=sequences,
                    max_sequence_size=5000)

    targets = np.zeros((len(sequences), len(label_ids)), dtype=np.int32)
    
    for i, (document_tags) in enumerate(tags):
        for tag in document_tags:
            if tag in label_ids:
                targets[i][label_ids[tag]] = 1
    
    del sequences, tags

    return samples, targets


# Load train dataset and count labels
#train_files = glob.glob(os.path.join(DATA_SET_DIR, "EURLEX57K", 'train', '*.json'))
train_files = sorted([os.path.join("data/datasets/EURLEX57K/train",doc) for doc in docs])
train_counts = Counter()
for filename in tqdm.tqdm(train_files):
    with open(filename) as file:
        data = json.load(file)
        for concept in data['concepts']:
            train_counts[concept] += 1

train_concepts = set(list(train_counts))

frequent, few = [], []
for i, (label, count) in enumerate(train_counts.items()):
    if count > 50:
        frequent.append(label)
    else:
        few.append(label)

# Load dev/test datasets and count labels
rest_files = glob.glob(os.path.join(DATA_SET_DIR, "EURLEX57K", 'dev', '*.json'))
rest_files += glob.glob(os.path.join(DATA_SET_DIR, "EURLEX57K", 'test', '*.json'))
rest_concepts = set()
for filename in tqdm.tqdm(rest_files):
    with open(filename) as file:
        data = json.load(file)
        for concept in data['concepts']:
            rest_concepts.add(concept)

# Load label descriptors
with open(os.path.join(DATA_SET_DIR, "EURLEX57K",
                        '{}.json'.format("EURLEX57K"))) as file:
    data = json.load(file)
    none = set(data.keys())

none = none.difference(train_concepts.union((rest_concepts)))
parents = []
for key, value in data.items():
    parents.extend(value['parents'])
none = none.intersection(set(parents))

# Compute zero-shot group
zero = list(rest_concepts.difference(train_concepts))
true_zero = deepcopy(zero)
zero = zero + list(none)


# Compute margins for frequent / few / zero groups
label_ids = dict()
margins = [(0, len(frequent)+len(few)+len(true_zero))]
k = 0
for group in [frequent, few, true_zero]:
    margins.append((k, k+len(group)))
    for concept in group:
        label_ids[concept] = k
        k += 1
margins[-1] = (margins[-1][0], len(frequent)+len(few)+len(true_zero))


documents = load_dataset_s(docs)
val_samples, val_tags = process_dataset(documents)
val_samples, val_targets = encode_dataset(val_samples, val_tags)
#print("val_samples", val_samples)
#print("val_targets", val_targets)

#for doc in docs:
#    sentences.append(data_loader_sim())

#print(f"Number of sentences: {len(sentences)}")

#for sentence in sentences[:10]:
"""
for sentence in documents[:10]:
    vect = vectorizer.vectorize_inputs(sequences=sentence)
    print(" ---------  Vectorised")
    #print(vect)

    predictions = model2.predict(vect)
    print("Raw preds:")
    print(predictions.dtype)
    print(predictions.size)
    print(predictions.shape)
    print("")

    pred_targets = (predictions > 0.7).astype('int32')
    print("Preds > .7 and int32:")
    print(pred_targets.dtype)
    print(pred_targets.size)
    print(pred_targets.shape)
    
    if True in pred_targets:
        print("True")
    index_true = np.where(pred_targets == True)
    print(f"# of True {len(index_true[1])}")
    print("")

    onlyTrues = np.where(predictions > 0.7)[0]
    print("Only Trues:")
    print(onlyTrues)
    print(onlyTrues.dtype)
    print(onlyTrues.size)
    print(onlyTrues.shape)
 

    #print(pred_targets)
"""
#vect = vectorizer.vectorize_inputs(sequences=sentences[:10])
#vect = vectorizer.vectorize_inputs(sequences=documents[:10])
print(" ---------  Vectorised")

predictions = model2.predict(val_samples)
print("Raw preds:")
print(predictions.dtype)
print(predictions.size)
print(predictions.shape)
print("")

threshold = 0.7

pred_targets = (predictions > threshold).astype('int32')
print(f"Preds > {threshold} and int32:")
print(pred_targets.dtype)
print(pred_targets.size)
print(pred_targets.shape)
"""
if True in pred_targets:
    print("True")
index_true = np.where(pred_targets == True)[0]
#print(f"# of True {len(index_true[1])}")
print(index_true)
print(index_true.dtype)
print(index_true.size)
print(index_true.shape)
print("")

onlyTrues = np.where(predictions > threshold)[0]
print("Only Trues:")
print(onlyTrues)
print(onlyTrues.dtype)
print(onlyTrues.size)
print(onlyTrues.shape)
"""
for _, sub in enumerate(pred_targets):
    #print(len(sub))
    ot = np.where(sub == True)
    #print(ot[0])
    #for lab in ot[0]
    print(ot[0])
    labelss = [eurlex[str(id_)]["label"] for id_ in ot[0]]
    print(sorted(docs)[_], labelss)