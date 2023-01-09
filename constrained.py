from collections import Counter
from datetime import datetime
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE
from keras import backend as K
from keras.layers import Layer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import tensorflow_addons as tfa

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shap
import sys
import tensorflow as tf
import winsound

import requests
from Bio import Entrez

import logging
from collections import Counter

from kerashypetune import KerasGridSearch, KerasGridSearchCV
from kerashypetune import KerasRandomSearch, KerasRandomSearchCV
from kerashypetune import KerasBayesianSearch, KerasBayesianSearchCV
from scipy import stats as stats
from hyperopt import hp, Trials

# Data under license and not provided for this repository
indt = pd.read_json("indata2.json")
# load imputed meta-data
train_meta = pd.read_csv("clean_Training.csv", index_col=0)
test_meta  = pd.read_csv("clean_Testing.csv", index_col=0)
val_meta   = pd.read_csv("clean_Validation.csv", index_col=0)
# remove baseline features
train_meta = train_meta.loc[ :, [x for x in train_meta.columns if x.find("_bl")==-1 and x.find("_BL")==-1]]
test_meta = test_meta.loc[ :, [x for x in test_meta.columns if x.find("_bl")==-1 and x.find("_BL")==-1]]
val_meta = val_meta.loc[ :, [x for x in val_meta.columns if x.find("_bl")==-1 and x.find("_BL")==-1]]
# convert categorical
PTGENDER = {'Female': 1, 'Male': 0}
PTETHCAT = {'Not Hisp/Latino': 0, 'Hisp/Latino': 1}
PTRACCAT = {'White': 0, 'Black': 1, 'Am Indian/Alaskan': 2, 'Hawaiian/Other PI': 3, 'Asian': 4, 'More than one': 5}
PTMARRY  = {'Married': 0, 'Divorced': 1, 'Widowed': 2, 'Never married': 3}
train_meta['PTGENDER']  = train_meta.apply( lambda row: PTGENDER[row['PTGENDER']], axis=1)
train_meta['PTETHCAT']  = train_meta.apply( lambda row: PTETHCAT[row['PTETHCAT']], axis=1)
train_meta['PTRACCAT']  = train_meta.apply( lambda row: PTRACCAT[row['PTRACCAT']], axis=1)
train_meta['PTMARRY']   = train_meta.apply( lambda row: PTMARRY [row['PTMARRY' ]], axis=1)
test_meta ['PTGENDER']  = test_meta.apply ( lambda row: PTGENDER[row['PTGENDER']], axis=1)
test_meta ['PTETHCAT']  = test_meta.apply ( lambda row: PTETHCAT[row['PTETHCAT']], axis=1)
test_meta ['PTRACCAT']  = test_meta.apply ( lambda row: PTRACCAT[row['PTRACCAT']], axis=1)
test_meta ['PTMARRY']   = test_meta.apply ( lambda row: PTMARRY [row['PTMARRY' ]], axis=1)
val_meta  ['PTGENDER']  = val_meta.apply  ( lambda row: PTGENDER[row['PTGENDER']], axis=1)
val_meta  ['PTETHCAT']  = val_meta.apply  ( lambda row: PTETHCAT[row['PTETHCAT']], axis=1)
val_meta  ['PTRACCAT']  = val_meta.apply  ( lambda row: PTRACCAT[row['PTRACCAT']], axis=1)
val_meta  ['PTMARRY']   = val_meta.apply  ( lambda row: PTMARRY [row['PTMARRY' ]], axis=1)
# convert y to numpy array
X_train, y_train = indt.loc[ train_meta.index, : ], train_meta.status.values
X_test, y_test   = indt.loc[ test_meta.index,  : ], test_meta.status.values
X_val, y_val     = indt.loc[ val_meta.index,   : ], val_meta.status.values
# merge after dropping status cols
X_train = pd.merge(X_train, train_meta, left_on=X_train.index, right_on=train_meta.index)
X_test  = pd.merge(X_test,  test_meta,  left_on=X_test.index,  right_on=test_meta.index )
X_val   = pd.merge(X_val,   val_meta,   left_on=X_val.index,   right_on=val_meta.index  )
# remove unnecessary columns
X_test.drop (['Unnamed: 86'], axis=1, inplace=True)
X_val.drop  (['Unnamed: 86'], axis=1, inplace=True)
# Data pre-split into training (56%), test (30%) and validation (14%)

# Reactome pathway data
# First layer
def get_pathway_gene_map() -> dict:
    pathway_gene_mapd = {}
    with open("./project_code3/data/ReactomePathways.gmt",'r') as f:
        for line in f:
            line = line.rstrip().split("\t")
            for gene in line[2:]:
                try:
                    pathway_gene_mapd[line[0]].update( {gene:1} )
                except KeyError:
                    pathway_gene_mapd[line[0]] = {gene:1}
    # Convert to dataframe and save
    pathway_gene_map = pd.DataFrame.from_dict(pathway_gene_mapd)
    pathway_gene_map.fillna(0,inplace=True)
    return pathway_gene_map

pathway_gene_l1 = get_pathway_gene_map()
pathway_names = pd.read_csv("./project_code3/data/ReactomePathways.txt", sep="\t", header=None)
pathway_names.columns = ['ID', 'name', 'species']
pathway_names = pathway_names[pathway_names.ID.str.contains('-HSA-')]

# Get constraints for SNP data
genes_snp_constraints = [x for x in pathway_gene_l1.index if x+"_x" in indt.columns[:11287]]
pathway_gene_l1_snps = pathway_gene_l1.loc[genes_snp_constraints,:]
colnames = []
for x in pathway_gene_l1_snps.columns:
    p = pathway_names[pathway_names.name==x]
    if len(p)>=1:
        colnames.append(p.iloc[0,:].ID)
    else:
        print(x)
        
pathway_gene_l1_snps.columns = colnames


# Gene expression
gene_exp_constraints = [x for x in pathway_gene_l1.index if x+"_y" in indt.columns[11287:21516]]
pathway_gene_l1_exp = pathway_gene_l1.loc[gene_exp_constraints,:]
colnames = []
for x in pathway_gene_l1_exp.columns:
    p = pathway_names[pathway_names.name==x]
    if len(p)>=1:
        colnames.append(p.iloc[0,:].ID)
    else:
        print(x)
        
pathway_gene_l1_exp.columns = colnames
# Convert pathway relation file data to constraints matrices for subsequent layers

def get_constraint_layers(df):
    constraints = {}
    ndx_to_remove = []
    children = np.unique(df.child)
    for ndx, row in df.iterrows():
        if str(row.parent) not in children:
            try:
                constraints[row.child].update( {row.parent:1} )
            except KeyError:
                constraints[row.child] = {row.parent:1}
            # parent-child pair for removal later
            ndx_to_remove.append(ndx)
    # remove ndx
    new_df = df[df.index.isin(ndx_to_remove)==False]
    # convert constraints to df
    constraints = pd.DataFrame.from_dict(constraints)
    constraints.fillna(0,inplace=True)
    return constraints, new_df

rel = pd.read_csv("./pathways/ReactomePathwaysRelation.txt", sep='\t')
rel.columns = ['child', 'parent']
rel = rel[rel.child.str.contains("R-HSA")] # human pathways

l1, new_df = get_constraint_layers(rel)
l2, new_df2 = get_constraint_layers(new_df)

l2 = l2.loc[:, l3.index]
l1 = l1.loc[:, l2.index]

temps = [x for x in pathway_gene_l1_snps.columns if x in l1.index]
tempe = [x for x in pathway_gene_l1_exp.columns if x in l1.index]

pathway_gene_l1_snps = pathway_gene_l1_snps.loc[:, temps]
pathway_gene_l1_exp = pathway_gene_l1_exp.loc[:, tempe]

cols = [x + "_x" for x in pathway_gene_l1_snps.index] + [x + "_y" for x in pathway_gene_l1_exp.index] + list(X_train.columns[21516:])

X_train = X_train.loc[:,cols]
X_test = X_test.loc[:,cols]
X_val = X_val.loc[:,cols]

# oversample training
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)

# Convert to OHE
y_resampled = tf.keras.utils.to_categorical( y_resampled )
y_train      = tf.keras.utils.to_categorical( y_train )
y_test      = tf.keras.utils.to_categorical( y_test )
y_val       = tf.keras.utils.to_categorical( y_val)

l1 = l1.loc[pathway_gene_l1_exp.columns,:]

scaler = MinMaxScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_val  = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Converted constraints to numpy arrays
pathway_gene_l1_snps = pathway_gene_l1_snps.values.astype('float32')
pathway_gene_l1_exp = pathway_gene_l1_exp.values.astype('float32')
l1 = l1.values.astype('float32')
l2 = l2.values.astype('float32')

# Model definition
num_snps = 10151 
num_expr = 10151
num_meta = 45

# SNPS
snps_input = keras.Input(shape=num_snps, name="Gene_SNP_info") # sparse input: SNP data
RPs1   = layers.Dense(1744, kernel_initializer=keras.initializers.Constant(pathway_gene_l1_snps), trainable=False, name="RP_snps1")(snps_input)
RPs2   = layers.Dense(412, kernel_initializer=keras.initializers.Constant(l1), trainable=False, name="RP_snps2")(RPs1)
RPs3   = layers.Dense(160, kernel_initializer=keras.initializers.Constant(l2), trainable=False, name="RP_snps3")(RPs2)
# additional dense layer
S4     = layers.Dense(40,activation="sigmoid")(RPs3)
# relu/sigmoid 
# expr
expr_input = keras.Input(shape=num_expr, name="expr_info")  # dense input: gene expr data
RPe1   = layers.Dense(1744, kernel_initializer=keras.initializers.Constant(pathway_gene_l1_exp), trainable=False, name="RP_exp1")(expr_input)
RPe2   = layers.Dense(412, kernel_initializer=keras.initializers.Constant(l1), trainable=False, name="RP_exp2")(RPe1)
RPe3   = layers.Dense(160, kernel_initializer=keras.initializers.Constant(l2), trainable=False, name="RP_exp3")(RPe2)
# additional dense layer
E4     = layers.Dense(40,activation="sigmoid")(RPe3)

# meta-info
meta_input = keras.Input(shape=num_meta, name="meta_info")
# lm1    = keras.layers.Dense(85)(meta_input)

## Concatenate and batch normalize
concat = keras.layers.concatenate([S4, E4, meta_input], axis=1)
bnorm  = keras.layers.BatchNormalization()(concat)

# hidden layers
l2_ = layers.Dense(21,name="Layer2", kernel_regularizer=regularizers.L1(0.009), activation='relu')(bnorm) 
d2 = layers.Dropout(0.597)(l2_)
l3_ = layers.Dense(3,name="Layer3", kernel_regularizer=regularizers.L1(0.001))(d2)
# Output layer
lo         = layers.Dense(3,name="Output",activation="softmax")(l3_) # l3_)
# Model
model      = keras.Model(
                    inputs  = [snps_input, expr_input, meta_input],
                    outputs = [lo],
)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.008,
                    decay_steps=17,
                    decay_rate=0.96,
)

model.compile(
    
                optimizer= keras.optimizers.Adam(learning_rate=lr_schedule, name="adam"),
                loss     = tf.keras.losses.CategoricalCrossentropy(),
                metrics  = [ 
                    tf.keras.metrics.CategoricalAccuracy(name='sp_acc'),
                    tf.keras.metrics.AUC(name='auc', ),
                    tf.keras.metrics.FalsePositives(name='fp'),
                    tf.keras.metrics.Precision(name='pre'),
                    tf.keras.metrics.Recall(name='rec'),
                ]
)

# After Bayes optimization, the model was trained and saved
# Load saved model
model = tf.keras.models.load_model('./Constrained/')
# Evaluate on test set
model.evaluate([X_test[:,:num_snps], X_test[:,num_snps:num_snps+num_expr], X_test[:,-1*num_meta:]],y_test)
