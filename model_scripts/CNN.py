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
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling1D, Conv1D, Reshape

import wandb
from wandb.keras import WandbCallback

LR, BATCH_SIZE, EPOCHS, INPUT_SHAPE_3_MER, INPUT_SHAPE_6_MER = 0.001, 128, 50, (125, 1), (15625, 1)

print('Packages loaded!')

####################################################################################################

# AMOUNT OF UNIQUE LABELS AT EACH TAXON LEVEL
fam_count, gen_count, spe_count = 349, 954, 1569

####################################################################################################

# LOADING ENCODED SEQUENCES for the CNN models in 2 processing variations
# FOR CNN  |  with 3-mer
x_train_CNN_na3 = np.load('arrays/CNN/x_train_CNN_na3.npy')
# x_train_CNN_a3 = np.load('arrays/CNN/x_train_CNN_a3.npy')
x_test_CNN_na3 = np.load('arrays/CNN/x_test_CNN_na3.npy')
dataval_CNN_na3 = np.load('arrays/CNN/dataval_CNN_na3.npy')
# dataval_CNN_a3 = np.load('arrays/CNN/dataval_CNN_a3.npy')
print('3-mers LOADED')
# -----------------------------------------------------------------------------
# FOR CNN  |  with 6-mer
x_train_CNN_na6 = np.load('arrays/CNN/x_train_CNN_na6.npy')
x_train_CNN_a6 = np.load('arrays/CNN/x_train_CNN_a6.npy')
x_test_CNN_na6 = np.load('arrays/CNN/x_test_CNN_na6.npy')
dataval_CNN_na6 = np.load('arrays/CNN/dataval_CNN_na6.npy')
dataval_CNN_a6 = np.load('arrays/CNN/dataval_CNN_a6.npy')
print('6-mers LOADED')
# -----------------------------------------------------------------------------
print('CNN sequences LOADED')

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

def make_CNNmodel(input_shape, out_len, name='CNN'):
    CNNmodel = keras.Sequential(
        [
            Reshape(target_shape = input_shape, input_shape = input_shape[:-1]),
            Conv1D(4, 15, input_shape=input_shape),
            Activation('relu'),
            MaxPooling1D(pool_size=2),

            Conv1D(8, 10),
            Activation('relu'),
            MaxPooling1D(pool_size=2),

            Conv1D(12, 5),
            Activation('relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),

            Flatten(),
            Dense(256),
            Activation('relu'),
            Dropout(0.4),

            Dense(out_len, activation='softmax')
        ], 
        name = name
    )
    return CNNmodel

####################################################################################################

# The CNN models are created, tailored to the different input (k-mer) and output (taxon) shapes. The input is determined by the k-mer used and the output shape is determined by the amount of unique taxon labels.
# for Family
CNN_fam_3 = make_CNNmodel(input_shape=INPUT_SHAPE_3_MER, out_len=fam_count, name='CNN_fam_3') # with 3-mer
CNN_fam_6 = make_CNNmodel(input_shape=INPUT_SHAPE_6_MER, out_len=fam_count, name='CNN_fam_6') # with 6-mer
# for Genus
CNN_gen_3 = make_CNNmodel(input_shape=INPUT_SHAPE_3_MER, out_len=gen_count, name='CNN_gen_3')
CNN_gen_6 = make_CNNmodel(input_shape=INPUT_SHAPE_6_MER, out_len=gen_count, name='CNN_gen_6')
# for Species
CNN_spe_3 = make_CNNmodel(input_shape=INPUT_SHAPE_3_MER, out_len=spe_count, name='CNN_spe_3')
CNN_spe_6 = make_CNNmodel(input_shape=INPUT_SHAPE_6_MER, out_len=spe_count, name='CNN_spe_6')

####################################################################################################

def train_and_evaluate_model(model, train_data, train_labels, validation_data, validation_labels, test_data, test_labels):
    wandb.init(project = 'Final Training', entity = 'bachelorprojectgroup9', name=model.name)

    print (f'Loading {model.name} model...')
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LR), metrics=['accuracy'])
    print(model.summary())

    print (f'Fitting {model.name} model...')
    start_time = time.time()
    history = model.fit(train_data, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data = (validation_data, validation_labels), callbacks=[WandbCallback()])
    time_taken = round(time.time() - start_time)
    # history object is saved and can later be destinguished using the model/train_data names
    np.save('history/{}_{}.npy'.format(model.name, argname('train_data')), history.history)
    
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
    # score metrics are saved and can later be destinguished using the model names
    score_dict.to_csv(f'scores/{model.name}_evaluation.csv', index=False)

    wandb.finish()
    return

####################################################################################################

# RUNNING CNN MODELS
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# running the model at genus level with both k-mers
train_and_evaluate_model(CNN_gen_3, x_train_CNN_na3, y_train_gen_na, dataval_CNN_na3, labelsval_gen_na, x_test_CNN_na3, y_test_gen_na)
train_and_evaluate_model(CNN_gen_6, x_train_CNN_na6, y_train_gen_na, dataval_CNN_na6, labelsval_gen_na, x_test_CNN_na6, y_test_gen_na)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# running the 6-mer model at family and species level
train_and_evaluate_model(CNN_fam_6, x_train_CNN_na6, y_train_fam_na, dataval_CNN_na6, labelsval_fam_na, x_test_CNN_na6, y_test_fam_na)
train_and_evaluate_model(CNN_spe_6, x_train_CNN_na6, y_train_spe_na, dataval_CNN_na6, labelsval_spe_na, x_test_CNN_na6, y_test_spe_na)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# running the 6-mer model at genus level on the augmented data
train_and_evaluate_model(CNN_gen_6, x_train_CNN_a6, y_train_gen_a, dataval_CNN_a6, labelsval_gen_a, x_test_CNN_na6, y_test_gen_na)