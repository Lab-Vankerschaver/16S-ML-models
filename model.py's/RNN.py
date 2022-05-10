# LOADING PACKAGES
import numpy as np
import pandas as pd
import sys
from varname import argname
import time
import string
import os
import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Masking, Dot, Add, BatchNormalization
from keras.layers import MaxPooling1D, AveragePooling1D, Conv1D, Reshape
from keras.layers import TimeDistributed, LSTM, Bidirectional

import wandb
from wandb.keras import WandbCallback

LR, BATCH_SIZE, EPOCHS, MAX_LEN, INPUT_SHAPE_RNN, INPUT_SHAPE_4_MER, INPUT_SHAPE_7_MER = 0.001, 8, 3, 2000, (2000, 4), (625, 1), (78125, 1)

print('Packages loaded!')

###############################################################################

# AMOUNT OF UNIQUE LABELS AT EACH TAXON LEVEL
fam_count, gen_count, spe_count = 349, 954, 1569

###############################################################################

# LOADING ENCODED SEQUENCES for the RNN models in 2 processing variations
# FOR RNN  |  with regular one-hot-encoding
x_train_RNN_na0 = np.load(f'arrays/CNN/x_train_RNN_na0.npy')
# x_train_RNN_a0 = np.load(f'arrays/CNN/x_train_RNN_a0.npy')
x_test_RNN_na0 = np.load(f'arrays/CNN/x_test_RNN_na0.npy')
dataval_RNN_na0 = np.load(f'arrays/CNN/dataval_RNN_na0.npy')
# dataval_RNN_a0 = np.load(f'arrays/CNN/dataval_RNN_a0.npy')
print('Regular one-hot-encoded sequences LOADED')
# -----------------------------------------------------------------------------
# FOR RNN  |  with matation rate adjusted one-hot-encoding
x_train_RNN_na1 = np.load(f'arrays/CNN/x_train_RNN_na1.npy')
x_train_RNN_a1 = np.load(f'arrays/CNN/x_train_RNN_a1.npy')
x_test_RNN_na1 = np.load(f'arrays/CNN/x_test_RNN_na1.npy')
dataval_RNN_na1 = np.load(f'arrays/CNN/dataval_RNN_na1.npy')
dataval_RNN_a1 = np.load(f'arrays/CNN/dataval_RNN_a1.npy')
print('Mutation rate adjusted one-hot-encoded sequences LOADED')
# -----------------------------------------------------------------------------
print('RNN sequences LOADED')

###############################################################################

# LOADING one-hot encoded labels at each taxon level
# -----------------------------------------------------------------------------
# LABELS AT FAMILY LEVEL
y_train_fam_na = np.load(f'arrays/CNN/y_train_fam_na.npy')
# y_train_fam_a = np.load(f'arrays/CNN/y_train_fam_a.npy')
y_test_fam_na = np.load(f'arrays/CNN/y_test_fam_na.npy')
labelsval_fam_na = np.load(f'arrays/CNN/labelsval_fam_na.npy')
# labelsval_fam_a = np.load(f'arrays/CNN/labelsval_fam_a.npy')
print('Family label arrays LOADED')
# -----------------------------------------------------------------------------
# LABELS AT GENUS LEVEL
y_train_gen_na = np.load(f'arrays/CNN/y_train_gen_na.npy')
y_train_gen_a = np.load(f'arrays/CNN/y_train_gen_a.npy')
y_test_gen_na = np.load(f'arrays/CNN/y_test_gen_na.npy')
labelsval_gen_na = np.load(f'arrays/CNN/labelsval_gen_na.npy')
labelsval_gen_a = np.load(f'arrays/CNN/labelsval_gen_a.npy')
print('Genus label arrays LOADED')
# -----------------------------------------------------------------------------
# LABELS AT SPECIES LEVEL
y_train_spe_na = np.load(f'arrays/CNN/y_train_spe_na.npy')
# y_train_spe_a = np.load(f'arrays/CNN/y_train_spe_a.npy')
y_test_spe_na = np.load(f'arrays/CNN/y_test_spe_na.npy')
labelsval_spe_na = np.load(f'arrays/CNN/labelsval_spe_na.npy')
# labelsval_spe_a = np.load(f'arrays/CNN/labelsval_spe_a.npy')
print('Species label arrays LOADED')

###############################################################################
# BiLSTM
def make_BiLSTMmodel(out_len, INPUT_SHAPE=INPUT_SHAPE_RNN, name='BiLSTM'):
    BiLSTMmodel = keras.Sequential(
        [
            Masking(mask_value=0., input_shape=INPUT_SHAPE_RNN),
            Bidirectional(LSTM(128, return_sequences=True), merge_mode='sum'),
            Dropout(0.5),
            AveragePooling1D(4),
            Bidirectional(LSTM(128), merge_mode='sum'),
            Dropout(0.5),
            Dense((out_len), activation='softmax'),
        ],
        name=name
    )
    return BiLSTMmodel

###############################################################################

# ConvBiLSTM
def make_ConvBiLSTMmodel(out_len, INPUT_SHAPE=INPUT_SHAPE_RNN, name='ConvBiLSTM'):
    ConvBiLSTMmodel = keras.Sequential(
        [
            Masking(mask_value=0., input_shape=INPUT_SHAPE),
                        
            Conv1D(128, 3),
            AveragePooling1D(),
            Dropout(0.2),

            Conv1D(128, 3),
            AveragePooling1D(),
            Dropout(0.2),

            Conv1D(128, 3, use_bias=True),
            AveragePooling1D(),
            Dropout(0.2),
            
            Bidirectional(LSTM(128, activation='tanh'), merge_mode='sum'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(out_len, activation='softmax')
        ], 
        name = name
    )
    return ConvBiLSTMmodel

###############################################################################

# Read2Pheno
## Conv & Res net layers
CONV_NET_nr, RES_NET_nr, NET_filters, NET_window = 2, 1, 64, 2
## extra Dropout layer (after Res block)
DROP_r, POOL_s = 0.2, 2
## BiLSTM layer
LSTM_nodes = 128
## attention Layers
ATT_layers,vATT_nodes = 1, 128
## fully connected layers
FC_layers, FC_nodes, FC_drop = 1, 128, 0.3

# BLOCK FUNCTIONS
def conv_net_block(X, n_cnn_filters=256, cnn_window=9, block_name='convblock'):
    '''
    convolutional block with a 1D convolutional layer, a batch norm layer followed by a relu activation.
    parameters:
        n_cnn_filters: number of output channels
        cnn_window: window size of the 1D convolutional layer
    '''
    X = Conv1D(n_cnn_filters, cnn_window, strides=1, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    return X

def res_net_block(X, n_cnn_filters=256, cnn_window=9, block_name='resblock'):
    '''
    residual net block accomplished by a few convolutional blocks.
    parameters:
        n_cnn_filters: number of output channels
        cnn_window: window size of the 1D convolutional layer
    '''
    X_identity = X
    # cnn0
    X = Conv1D(n_cnn_filters, cnn_window, strides=1, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # cnn1
    X = Conv1D(n_cnn_filters, cnn_window, strides=1, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # cnn2
    X = Conv1D(n_cnn_filters, cnn_window, strides=1, padding='same')(X)
    X = BatchNormalization()(X)
    X = Add()([X, X_identity])
    X = Activation('relu')(X)
    return X

def attention_layer(H_lstm, n_layer, n_node, block_name='att'):
    '''
    feedforward attention layer accomplished by time distributed dense layers.
    parameters:
        n_layer: number of hidden layers
        n_node: number of hidden nodes
    '''
    H_emb = H_lstm
    for i in range(n_layer):
        H_lstm = TimeDistributed(Dense(n_node, activation="tanh"))(H_lstm)
    M = TimeDistributed(Dense(1, activation="linear"))(H_lstm)
    alpha = keras.layers.Softmax(axis=1)(M)
    r_emb = Dot(axes = 1)([alpha, H_emb])
    r_emb = Flatten()(r_emb)
    return r_emb

def fully_connected(r_emb, n_layer, n_node, drop_out_rate=0.5, block_name='fc'):
    '''
    fully_connected layer consists of a few dense layers.
    parameters:
        n_layer: number of hidden layers
        n_node: number of hidden nodes
        drop_out_rate: dropout rate to prevent the model from overfitting
    '''
    for i in range(n_layer):
        r_emb = Dense(n_node, activation="relu")(r_emb)
    r_emb = Dropout(drop_out_rate)(r_emb) 
    return r_emb
    
# TOTAL MODEL FUNCTION

def make_R2Pmodel(out_len, INPUT_SHAPE=INPUT_SHAPE_RNN, name='Read2Pheno'):
    X = Input(shape=INPUT_SHAPE)
    X_mask = Masking(mask_value=0.)(X)

    ## CONV Layers
    X_cnn = X_mask
    # conv_net
    for i in range(CONV_NET_nr):
        X_cnn = conv_net_block(X_cnn, n_cnn_filters=NET_filters, cnn_window=NET_window)
    # res_net
    for i in range(RES_NET_nr):
        X_cnn = res_net_block(X_cnn, n_cnn_filters=NET_filters, cnn_window=NET_window)

    ## Extra Pooling layer and Dropout
    X_pool = AveragePooling1D(pool_size=POOL_s)(X_cnn)
    X_drop = Dropout(DROP_r)(X_pool)

    ## RNN Layers
    H_lstm = Bidirectional(LSTM(LSTM_nodes, return_sequences=True), merge_mode='sum')(X_drop)
    H_lstm = Activation('tanh')(H_lstm)

    ## ATT Layers
    r_emb = attention_layer(H_lstm, n_layer=ATT_layers, n_node=ATT_nodes, block_name = 'att')
        
    # Fully connected layers
    r_emb = fully_connected(r_emb, n_layer=FC_layers, n_node=FC_nodes, drop_out_rate=FC_drop, block_name = 'fc')

    # Compile model
    out = Dense(out_len, activation='softmax', name='final_dense')(r_emb)
    R2Pmodel = Model(inputs = X, outputs = out, name = name)
    
    return R2Pmodel

###############################################################################

# The RNN models are created, tailored to the different output shapes. The output shape is determined by the amount of unique taxon labels.
# for Family
BiLSTM_fam = make_BiLSTMmodel(out_len=fam_count, name='BiLSTM_Family-level')
ConvBiLSTM_fam = make_ConvBiLSTMmodel(out_len=fam_count, name='ConvBiLSTM_Family-level')
R2P_fam = make_R2Pmodel(out_len=fam_count, name='Read2Pheno_Family-level')
# for Genus
BiLSTM_gen = make_BiLSTMmodel(out_len=gen_count, name='BiLSTM_Genus-level')
ConvBiLSTM_gen = make_ConvBiLSTMmodel(out_len=gen_count, name='ConvBiLSTM_Genus-level')
R2P_gen = make_R2Pmodel(out_len=gen_count, name='Read2Pheno_Genus-level')
# for Species
BiLSTM_spe = make_BiLSTMmodel(out_len=spe_count, name='BiLSTM_Species-level')
ConvBiLSTM_spe = make_ConvBiLSTMmodel(out_len=spe_count, name='ConvBiLSTM_Species-level')
R2P_spe = make_R2Pmodel(out_len=spe_count, name='Read2Pheno_Species-level')

###############################################################################

def train_and_evaluate_model(model, train_data, train_labels, validation_data, validation_labels, test_data, test_labels):
    wandb.init(project = 'Test training2', entity = 'bachelorprojectgroup9', name=model.name)

    print (f'Loading {model.name} model...')
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LR), metrics=['accuracy'])
    print(model.summary())

    print (f'Fitting {model.name} model...')
    start_time = time.time()
    history = model.fit(train_data, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data = (validation_data, validation_labels), callbacks=[WandbCallback()])
    time_taken = round(time.time() - start_time)
    np.save(f'history/{model.name}_history.npy', history.history)
    
    print (f'Evaluating {model.name} model...')
    test_labels_arg = np.argmax(test_labels, axis=1)
    test_predictions = np.argmax(model.predict(test_data), axis=1)
    loss, accuracy = model.evaluate(test_data, test_labels)

    # F1-score: harmonic mean of the precision and recall
    #   score from 0 to 1
    f1 = f1_score(y_true=test_labels_arg, y_pred=test_predictions, average='weighted')
    # Matthews correlation coefficient: coefficient of +1 represents a perfect prediction,
    #   0 an average random prediction and -1 an inverse prediction
    mcc = matthews_corrcoef(y_true=test_labels_arg, y_pred=test_predictions)

    score_dict = pd.DataFrame({'Model/run' : model.name, 'Data' : argname('train_data'), 'Training time' : time_taken, 'Test loss' : loss, 'Test accuracy' : accuracy, 'F1-score' : f1, 'MCC' : mcc}, index=[0])
    print(score_dict)
    score_dict.to_csv(f'scores/{model.name}_evaluation', index=False)

    wandb.finish()
    return history, score_dict

###############################################################################

# RUNNING RNN MODELS
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# running the models at genus level with both one-hot-encodings
BiLSTM_gen_0_history, BiLSTM_gen_0_score_dict = train_and_evaluate_model(BiLSTM_gen, x_train_RNN_na0, y_train_gen_na, dataval_RNN_na0, labelsval_gen_na, x_test_RNN_na0, y_test_gen_na)
ConvBiLSTM_gen_0_history, ConvBiLSTM_gen_0_score_dict = train_and_evaluate_model(ConvBiLSTM_gen, x_train_RNN_na0, y_train_gen_na, dataval_RNN_na0, labelsval_gen_na, x_test_RNN_na0, y_test_gen_na)
R2P_gen_0_history, R2P_gen_0_score_dict = train_and_evaluate_model(R2P_gen, x_train_RNN_na0, y_train_gen_na, dataval_RNN_na0, labelsval_gen_na, x_test_RNN_na0, y_test_gen_na)

BiLSTM_gen_history, BiLSTM_gen_score_dict = train_and_evaluate_model(BiLSTM_gen, x_train_RNN_na1, y_train_gen_na, dataval_RNN_na1, labelsval_gen_na, x_test_RNN_na1, y_test_gen_na)
ConvBiLSTM_gen_history, ConvBiLSTM_gen_score_dict = train_and_evaluate_model(ConvBiLSTM_gen, x_train_RNN_na1, y_train_gen_na, dataval_RNN_na1, labelsval_gen_na, x_test_RNN_na1, y_test_gen_na)
R2P_gen_history, R2P_gen_score_dict = train_and_evaluate_model(R2P_gen, x_train_RNN_na1, y_train_gen_na, dataval_RNN_na1, labelsval_gen_na, x_test_RNN_na1, y_test_gen_na)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# running the models at family and species level with the mutation rate adjusted one-hot-encoding
BiLSTM_fam_history, BiLSTM_fam_score_dict = train_and_evaluate_model(BiLSTM_fam, x_train_RNN_na1, y_train_fam_na, dataval_RNN_na1, labelsval_fam_na, x_test_RNN_na1, y_test_fam_na)
ConvBiLSTM_fam_history, ConvBiLSTM_fam_score_dict = train_and_evaluate_model(ConvBiLSTM_fam, x_train_RNN_na1, y_train_fam_na, dataval_RNN_na1, labelsval_fam_na, x_test_RNN_na1, y_test_fam_na)
R2P_fam_history, R2P_fam_score_dict = train_and_evaluate_model(R2P_fam, x_train_RNN_na1, y_train_fam_na, dataval_RNN_na1, labelsval_fam_na, x_test_RNN_na1, y_test_fam_na)

BiLSTM_spe_history, BiLSTM_spe_score_dict = train_and_evaluate_model(BiLSTM_spe, x_train_RNN_na1, y_train_spe_na, dataval_RNN_na1, labelsval_spe_na, x_test_RNN_na1, y_test_spe_na)
ConvBiLSTM_spe_history, ConvBiLSTM_spe_score_dict = train_and_evaluate_model(ConvBiLSTM_spe, x_train_RNN_na1, y_train_spe_na, dataval_RNN_na1, labelsval_spe_na, x_test_RNN_na1, y_test_spe_na)
R2P_spe_history, R2P_spe_score_dict = train_and_evaluate_model(R2P_spe, x_train_RNN_na1, y_train_spe_na, dataval_RNN_na1, labelsval_spe_na, x_test_RNN_na1, y_test_spe_na)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# running the models at family level with the mutation rate adjusted one-hot-encoding on the augmented data
BiLSTM_gen_a_history, BiLSTM_gen_a_score_dict = train_and_evaluate_model(BiLSTM_gen, x_train_RNN_a1, y_train_gen_a, dataval_RNN_a1, labelsval_gen_a, x_test_RNN_na1, y_test_gen_na)
ConvBiLSTM_gen_a_history, ConvBiLSTM_gen_a_score_dict = train_and_evaluate_model(ConvBiLSTM_gen, x_train_RNN_a1, y_train_gen_a, dataval_RNN_a1, labelsval_gen_a, x_test_RNN_na1, y_test_gen_na)
R2P_gen_a_history, R2P_gen_a_score_dict = train_and_evaluate_model(R2P_gen, x_train_RNN_a1, y_train_gen_a, dataval_RNN_a1, labelsval_gen_a, x_test_RNN_na1, y_test_gen_na)
