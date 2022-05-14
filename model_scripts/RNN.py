# LOADING PACKAGES
import numpy as np
import pandas as pd
from varname import argname
import time
import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Masking, Dot, Add, BatchNormalization
from keras.layers import AveragePooling1D, Conv1D
from keras.layers import TimeDistributed, LSTM, Bidirectional

import wandb
from wandb.keras import WandbCallback

LR, BATCH_SIZE, EPOCHS, = 0.001, 128, 60
MAX_LEN, MAX_LEN_V, INPUT_SHAPE_RNN, INPUT_SHAPE_RNN_V, = 2000, 500, (2000, 4), (500, 4)    # cut-off length

print('Packages loaded!')

####################################################################################################

# AMOUNT OF UNIQUE LABELS AT EACH TAXON LEVEL
fam_count, gen_count, spe_count = 349, 954, 1569

####################################################################################################

# LOADING ENCODED SEQUENCES for the RNN models in 2 processing variations
# FOR RNN  |  with regular one-hot-encoding
x_train_RNN_na0 = np.load('arrays/RNN/x_train_RNN_na0.npy')
# x_train_RNN_a0 = np.load('arrays/RNN/x_train_RNN_a0.npy')
x_test_RNN_na0 = np.load('arrays/RNN/x_test_RNN_na0.npy')
dataval_RNN_na0 = np.load('arrays/RNN/dataval_RNN_na0.npy')
# dataval_RNN_a0 = np.load('arrays/RNN/dataval_RNN_a0.npy')
print('Regular one-hot-encoded sequences LOADED')
# -----------------------------------------------------------------------------
# FOR RNN  |  with matation rate adjusted one-hot-encoding
x_train_RNN_na1 = np.load('arrays/RNN/x_train_RNN_na1.npy')
x_train_RNN_a1 = np.load('arrays/RNN/x_train_RNN_a1.npy')
x_test_RNN_na1 = np.load('arrays/RNN/x_test_RNN_na1.npy')
dataval_RNN_na1 = np.load('arrays/RNN/dataval_RNN_na1.npy')
dataval_RNN_a1 = np.load('arrays/RNN/dataval_RNN_a1.npy')
print('Mutation rate adjusted one-hot-encoded sequences LOADED')
# -----------------------------------------------------------------------------
# FOR RNN  |  with matation rate adjusted encoding on V-region selected data (CURRENTLY NOT USED)
# x_train_RNN_na1V = np.load('arrays/RNN/x_train_RNN_na1V.npy')
# x_train_RNN_a1V = np.load('arrays/RNN/x_train_RNN_a1V.npy')
# x_test_RNN_na1V = np.load('arrays/RNN/x_test_RNN_na1V.npy')
# dataval_RNN_na1V = np.load('arrays/RNN/dataval_RNN_na1V.npy')
# dataval_RNN_a1V = np.load('arrays/RNN/dataval_RNN_a1V.npy')
# print('Mutation rate adjusted one-hot-encoded V-region sequences LOADED')
# -----------------------------------------------------------------------------
print('RNN sequences LOADED')

####################################################################################################

# LOADING one-hot encoded labels at each taxon level
# -----------------------------------------------------------------------------
# LABELS AT FAMILY LEVEL
y_train_fam_na = np.load('arrays/family/y_train_fam_na.npy')
# y_train_fam_a = np.load('arrays/family/y_train_fam_a.npy')
y_test_fam_na = np.load('arrays/family/y_test_fam_na.npy')
labelsval_fam_na = np.load('arrays/family/labelsval_fam_na.npy')
# labelsval_fam_a = np.load('arrays/family/labelsval_fam_a.npy')
print('Family label arrays LOADED')
# -----------------------------------------------------------------------------
# LABELS AT GENUS LEVEL
y_train_gen_na = np.load('arrays/genus/y_train_gen_na.npy')
y_train_gen_a = np.load('arrays/genus/y_train_gen_a.npy')
y_test_gen_na = np.load('arrays/genus/y_test_gen_na.npy')
labelsval_gen_na = np.load('arrays/genus/labelsval_gen_na.npy')
labelsval_gen_a = np.load('arrays/genus/labelsval_gen_a.npy')
print('Genus label arrays LOADED')
# -----------------------------------------------------------------------------
# LABELS AT SPECIES LEVEL
y_train_spe_na = np.load('arrays/species/y_train_spe_na.npy')
# y_train_spe_a = np.load('arrays/species/y_train_spe_a.npy')
y_test_spe_na = np.load('arrays/species/y_test_spe_na.npy')
labelsval_spe_na = np.load('arrays/species/labelsval_spe_na.npy')
# labelsval_spe_a = np.load('arrays/species/labelsval_spe_a.npy')
print('Species label arrays LOADED')

####################################################################################################

# BiLSTM
def make_BiLSTMmodel(input_shape, out_len, name = 'BiLSTM'):
    BiLSTMmodel = keras.Sequential(
        [
            Masking(mask_value = 0., input_shape = input_shape),
            
            Bidirectional(LSTM(128, return_sequences = True), merge_mode = 'sum'),
            Dropout(0.5),

            AveragePooling1D(4),
            Bidirectional(LSTM(128), merge_mode = 'sum'),
            Dropout(0.5),

            Dense((out_len), activation = 'softmax'),
        ],
        name = name
    )
    return BiLSTMmodel

####################################################################################################

# ConvBiLSTM
def make_ConvBiLSTMmodel(input_shape, out_len, name = 'ConvBiLSTM'):
    ConvBiLSTMmodel = keras.Sequential(
        [
            Masking(mask_value = 0., input_shape = input_shape),
                        
            Conv1D(128, 3, padding = 'same'),
            AveragePooling1D(),

            Conv1D(128, 3, padding = 'same'),
            AveragePooling1D(),

            Conv1D(128, 3, padding = 'same', use_bias = True),
            AveragePooling1D(),
            Dropout(0.4),
            
            Bidirectional(LSTM(128, activation = 'tanh'), merge_mode = 'sum'),
            Dropout(0.2),
            
            Dense(128, activation = 'relu'),
            Dropout(0.2),
            Dense(out_len, activation = 'softmax')
        ], 
        name = name
    )
    return ConvBiLSTMmodel

####################################################################################################

# Read2Pheno
## Conv & Res net layers
CONV_NET_nr, RES_NET_nr, NET_filters, NET_window = 2, 1, 64, 2
## extra Dropout layer (after Res block)
DROP_r, POOL_s = 0.2, 2
## BiLSTM layer
LSTM_nodes, MERGE_m = 128, 'sum'
## attention Layers
ATT_layers, ATT_nodes = 1, 128
## fully connected layers
FC_layers, FC_nodes, FC_drop = 1, 128, 0.3

#####################################################################################################
# BLOCK FUNCTIONS
def conv_net_block(X, n_cnn_filters = 256, cnn_window = 9, block_name = 'convblock'):
    '''
    convolutional block with a 1D convolutional layer, a batch norm layer followed by a relu activation.
    parameters:
        n_cnn_filters: number of output channels
        cnn_window: window size of the 1D convolutional layer
    '''
    X = Conv1D(n_cnn_filters, cnn_window, strides = 1, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    return X

def res_net_block(X, n_cnn_filters = 256, cnn_window = 9, block_name = 'resblock'):
    '''
    residual net block accomplished by a few convolutional blocks.
    parameters:
        n_cnn_filters: number of output channels
        cnn_window: window size of the 1D convolutional layer
    '''
    X_identity = X
    # cnn0
    X = Conv1D(n_cnn_filters, cnn_window, strides = 1, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # cnn1
    X = Conv1D(n_cnn_filters, cnn_window, strides = 1, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # cnn2
    X = Conv1D(n_cnn_filters, cnn_window, strides = 1, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Add()([X, X_identity])
    X = Activation('relu')(X)
    return X

def attention_layer(H_lstm, n_layer, n_node, block_name = 'att'):
    '''
    feedforward attention layer accomplished by time distributed dense layers.
    parameters:
        n_layer: number of hidden layers
        n_node: number of hidden nodes
    '''
    H_emb = H_lstm
    for i in range(n_layer):
        H_lstm = TimeDistributed(Dense(n_node, activation = "tanh"))(H_lstm)
    M = TimeDistributed(Dense(1, activation = "linear"))(H_lstm)
    alpha = keras.layers.Softmax(axis = 1)(M)
    r_emb = Dot(axes = 1)([alpha, H_emb])
    r_emb = Flatten()(r_emb)
    return r_emb

def fully_connected(r_emb, n_layer, n_node, drop_out_rate = 0.5, block_name = 'fc'):
    '''
    fully_connected layer consists of a few dense layers.
    parameters:
        n_layer: number of hidden layers
        n_node: number of hidden nodes
        drop_out_rate: dropout rate to prevent the model from overfitting
    '''
    for i in range(n_layer):
        r_emb = Dense(n_node, activation = "relu")(r_emb)
    r_emb = Dropout(drop_out_rate)(r_emb) 
    return r_emb
    
#####################################################################################################
# MODEL COMPILING FUNCTION

def make_R2Pmodel(input_shape, out_len, name = 'Read2Pheno'):
    X = Input(shape = input_shape)
    X_mask = Masking(mask_value = 0.)(X)

    ## CONV Layers
    X_cnn = X_mask
    # conv_net
    for i in range(CONV_NET_nr):
        X_cnn = conv_net_block(
            X_cnn, 
            n_cnn_filters = NET_filters, 
            cnn_window = NET_window
            )
    # res_net
    for i in range(RES_NET_nr):
        X_cnn = res_net_block(
            X_cnn, 
            n_cnn_filters = NET_filters, 
            cnn_window = NET_window
            )

    ## Extra Pooling layer and Dropout
    X_pool = AveragePooling1D(pool_size = POOL_s)(X_cnn)
    X_drop = Dropout(DROP_r)(X_pool)

    ## RNN Layers
    H_lstm = Bidirectional(LSTM(LSTM_nodes, return_sequences = True), merge_mode = MERGE_m)(X_drop)
    H_lstm = Activation('tanh')(H_lstm)

    ## ATT Layers
    r_emb = attention_layer(
        H_lstm, 
        n_layer = ATT_layers, 
        n_node = ATT_nodes, 
        block_name = 'att'
        )    
    # Fully connected layers
    r_emb = fully_connected(
        r_emb, 
        n_layer = FC_layers, 
        n_node = FC_nodes, 
        drop_out_rate = FC_drop, 
        block_name = 'fc'
        )

    # Compile model
    out = Dense(out_len, activation = 'softmax', name = 'final_dense')(r_emb)
    R2Pmodel = Model(inputs = X, outputs = out, name = name)
    
    return R2Pmodel

####################################################################################################

# Create RNN models
# ----------------------------------------------------------
# for Family (output length = number of unique labels)
BiLSTM_fam = make_BiLSTMmodel(
    input_shape = INPUT_SHAPE_RNN,
    out_len = fam_count, 
    name = 'BiLSTM_fam'
    )
ConvBiLSTM_fam = make_ConvBiLSTMmodel(
    input_shape = INPUT_SHAPE_RNN,
    out_len = fam_count
    , name = 'ConvBiLSTM_fam'
    )
R2P_fam = make_R2Pmodel(
    input_shape = INPUT_SHAPE_RNN,
    out_len = fam_count, 
    name = 'R2P_fam'
    )
# ----------------------------------------------------------
# for Genus
BiLSTM_gen = make_BiLSTMmodel(
    input_shape = INPUT_SHAPE_RNN,
    out_len = gen_count, 
    name = 'BiLSTM_gen'
    )
ConvBiLSTM_gen = make_ConvBiLSTMmodel(
    input_shape = INPUT_SHAPE_RNN,
    out_len = gen_count, 
    name = 'ConvBiLSTM_gen'
    )
R2P_gen = make_R2Pmodel(
    input_shape = INPUT_SHAPE_RNN,
    out_len = gen_count, 
    name='R2P_gen'
    )
# ----------------------------------------------------------
# for Species
BiLSTM_spe = make_BiLSTMmodel(
    input_shape = INPUT_SHAPE_RNN,
    out_len = spe_count, 
    name = 'BiLSTM_spe'
    )
ConvBiLSTM_spe = make_ConvBiLSTMmodel(
    input_shape = INPUT_SHAPE_RNN,
    out_len = spe_count, 
    name = 'ConvBiLSTM_spe'
    )
R2P_spe = make_R2Pmodel(
    input_shape = INPUT_SHAPE_RNN,
    out_len = spe_count, 
    name='R2P_spe'
    )
# ----------------------------------------------------------
# ----------------------------------------------------------
# for Genus with V-region selected data (CURRENLTY NOT USED)
# BiLSTM_gen_V = make_BiLSTMmodel(
#     input_shape = INPUT_SHAPE_RNN_V,
#     out_len = gen_count, 
#     name = 'BiLSTM_gen_V'
#     )
# ConvBiLSTM_gen_V = make_ConvBiLSTMmodel(
#     input_shape = INPUT_SHAPE_RNN_V,
#     out_len = gen_count, 
#     name = 'ConvBiLSTM_gen_V'
#     )
# R2P_gen_V = make_R2Pmodel(
#     input_shape = INPUT_SHAPE_RNN_V,
#     out_len = gen_count, 
#     name='R2P_gen_V'
#     )

####################################################################################################

def train_and_evaluate_model(
    model, 
    train_data, train_labels, 
    validation_data, validation_labels, 
    test_data, test_labels
    ):

    wandb.init(project = 'Final Training', entity = 'bachelorprojectgroup9', name = model.name)

    # LOADING MODEL
    print ('Loading {} model...'.format(model.name))
    model.compile(
        loss = 'categorical_crossentropy', 
        optimizer = Adam(learning_rate = LR), 
        metrics = ['accuracy']
        )
    print(model.summary())

    # FITTING MODEL
    print ('Fitting {} model...'.format(model.name))
    start_time = time.time()
    history = model.fit(
        train_data, train_labels, 
        epochs = EPOCHS, batch_size = BATCH_SIZE, 
        validation_data = (validation_data, validation_labels), 
        callbacks = [WandbCallback()]
        )
    time_taken = round(time.time() - start_time)
    # history object is saved and can later be destinguished using the model/train_data names
    np.save('history/{}_{}.npy'.format(model.name, argname('train_data')), history.history)
    
    # EVALUATING MODEL
    print ('Evaluating {} model...'.format(model.name))
    test_labels_arg = np.argmax(test_labels, axis = 1)
    test_predictions = np.argmax(model.predict(test_data), axis = 1)
    loss, accuracy = model.evaluate(test_data, test_labels)
    # F1-score: harmonic mean of the precision and recall
    #   score from 0 to 1
    f1 = f1_score(y_true = test_labels_arg, y_pred = test_predictions, average = 'weighted')
    # Matthews correlation coefficient: coefficient of +1 represents a perfect prediction,
    #   0 an average random prediction and -1 an inverse prediction
    mcc = matthews_corrcoef(y_true = test_labels_arg, y_pred = test_predictions)

    score_dict = pd.DataFrame({
        'Model/run' : model.name, 
        'Data' : argname('train_data'), 
        'Training time' : time_taken, 
        'Test loss' : loss, 
        'Test accuracy' : accuracy, 
        'F1-score' : f1, 
        'MCC' : mcc}, 
        index = [0]
        )
    print(score_dict)
    # score metrics are saved and can later be destinguished using the model names
    score_dict.to_csv('scores/{}_evaluation.csv'.format(model.name), index = False)

    wandb.finish()
    return

####################################################################################################

# RUNNING RNN MODELS
# ----------------------------------------------------------
# ----------------------------------------------------------
# running the RNN models at genus level 
# with regualar and mutation rate adjusted one-hot-encoding
# on the non-augmented data
train_and_evaluate_model(
    BiLSTM_gen, 
    x_train_RNN_na0, y_train_gen_na, 
    dataval_RNN_na0, labelsval_gen_na, 
    x_test_RNN_na0, y_test_gen_na
    )
train_and_evaluate_model(
    ConvBiLSTM_gen, 
    x_train_RNN_na0, y_train_gen_na, 
    dataval_RNN_na0, labelsval_gen_na, 
    x_test_RNN_na0, y_test_gen_na
    )
train_and_evaluate_model(
    R2P_gen, 
    x_train_RNN_na0, y_train_gen_na, 
    dataval_RNN_na0, labelsval_gen_na, 
    x_test_RNN_na0, y_test_gen_na
    )
# ----------------------------------------------------------
train_and_evaluate_model(
    BiLSTM_gen, 
    x_train_RNN_na1, y_train_gen_na, 
    dataval_RNN_na1, labelsval_gen_na, 
    x_test_RNN_na1, y_test_gen_na
    )
train_and_evaluate_model(
    ConvBiLSTM_gen, 
    x_train_RNN_na1, y_train_gen_na, 
    dataval_RNN_na1, labelsval_gen_na, 
    x_test_RNN_na1, y_test_gen_na
    )
train_and_evaluate_model(
    R2P_gen, 
    x_train_RNN_na1, y_train_gen_na, 
    dataval_RNN_na1, labelsval_gen_na, 
    x_test_RNN_na1, y_test_gen_na
    )
# ----------------------------------------------------------
# ----------------------------------------------------------
# running the RNN models at family and species level 
# with the mutation rate adjusted one-hot-encoding
# on the non-augmented data
train_and_evaluate_model(
    BiLSTM_fam, 
    x_train_RNN_na1, y_train_fam_na, 
    dataval_RNN_na1, labelsval_fam_na, 
    x_test_RNN_na1, y_test_fam_na
    )
train_and_evaluate_model(
    ConvBiLSTM_fam, 
    x_train_RNN_na1, y_train_fam_na, 
    dataval_RNN_na1, labelsval_fam_na, 
    x_test_RNN_na1, y_test_fam_na
    )
train_and_evaluate_model(
    R2P_fam, 
    x_train_RNN_na1, y_train_fam_na, 
    dataval_RNN_na1, labelsval_fam_na, 
    x_test_RNN_na1, y_test_fam_na
    )
# ----------------------------------------------------------
train_and_evaluate_model(
    BiLSTM_spe, 
    x_train_RNN_na1, y_train_spe_na, 
    dataval_RNN_na1, labelsval_spe_na, 
    x_test_RNN_na1, y_test_spe_na
    )
train_and_evaluate_model(
    ConvBiLSTM_spe, 
    x_train_RNN_na1, y_train_spe_na, 
    dataval_RNN_na1, labelsval_spe_na, 
    x_test_RNN_na1, y_test_spe_na
    )
train_and_evaluate_model(
    R2P_spe, 
    x_train_RNN_na1, y_train_spe_na, 
    dataval_RNN_na1, labelsval_spe_na, 
    x_test_RNN_na1, y_test_spe_na
    )
# ----------------------------------------------------------
# ----------------------------------------------------------
# running the RNN models at genus level 
# with the mutation rate adjusted one-hot-encoding 
# on the augmented data
train_and_evaluate_model(
    BiLSTM_gen, 
    x_train_RNN_a1, y_train_gen_a, 
    dataval_RNN_a1, labelsval_gen_a, 
    x_test_RNN_na1, y_test_gen_na
    )
train_and_evaluate_model(
    ConvBiLSTM_gen, 
    x_train_RNN_a1, y_train_gen_a, 
    dataval_RNN_a1, labelsval_gen_a, 
    x_test_RNN_na1, y_test_gen_na
    )
train_and_evaluate_model(
    R2P_gen, 
    x_train_RNN_a1, y_train_gen_a, 
    dataval_RNN_a1, labelsval_gen_a, 
    x_test_RNN_na1, y_test_gen_na
    )
# ----------------------------------------------------------
# ----------------------------------------------------------
# running the RNN models at genus level 
# with the mutation rate adjusted one-hot-encoding 
# on the V-region selected data (CURRENLY NOT USED)
# train_and_evaluate_model(
#     BiLSTM_gen_V, 
#     x_train_RNN_na1V, y_train_gen_na, 
#     dataval_RNN_na1V, labelsval_gen_na, 
#     x_test_RNN_na1V, y_test_gen_na
#     )
# train_and_evaluate_model(
#     ConvBiLSTM_gen_V, 
#     x_train_RNN_na1V, y_train_gen_na, 
#     dataval_RNN_na1V, labelsval_gen_na, 
#     x_test_RNN_na1V, y_test_gen_na
#     )
# train_and_evaluate_model(
#     R2P_gen_V, 
#     x_train_RNN_na1V, y_train_gen_na, 
#     dataval_RNN_na1V, labelsval_gen_na, 
#     x_test_RNN_na1V, y_test_gen_na
#     )

####################################################################################################

print('RNN training complete!')