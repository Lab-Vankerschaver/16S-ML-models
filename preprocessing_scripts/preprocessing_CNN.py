# LOADING PACKAGES

import numpy as np
import pandas as pd
from itertools import product

print('Packages loaded!')

# LOADING THE NON-AUGMENTED AND AUGMENTED DATASETS

train_na = pd.read_csv('df_train_0.csv')
val_na = pd.read_csv('df_val_0.csv')
test_na = pd.read_csv('df_test_0.csv')

train_a = pd.read_csv('df_train_1.csv')
val_a = pd.read_csv('df_val_1.csv')

print('Datasets loaded!')

# Generating k-mers
# For use in the Convolutional Neural Networks (CNN), the sequences are processed into a frequency table of k-mers.

# define all possible k-mers
alphabet = "AGTCN"
four_mers = [''.join(chars) for chars in product(*(4*(alphabet,)))]
seven_mers = [''.join(chars) for chars in product(*(7*(alphabet,)))]

def one_hot_k(sequence, kmers):
    # k-mer length
    k = len(kmers[0])
    # initialise k-mer dictionary with all possible k-mers as keys and count (0) as values
    kmer_dict = dict.fromkeys(kmers, 0)

    # standardise the sequence
    # by replacing U with T and all ambiguous bases with N
    sequence = sequence.replace('U', 'T').replace('Y', 'N').replace('R', 'N').replace('W', 'N').replace('S', 'N').replace('K', 'N').replace('M', 'N').replace('D', 'N').replace('V', 'N').replace('H', 'N').replace('B', 'N').replace('X', 'N').replace('-', 'N')
    # count every k-mer in the sequence
    for i in range(0, len(sequence) -k+1):
        kmer_dict[sequence[i:i+k]] += 1

    # k-mer frequency array from the dictionary values
    k_array = np.array(list(kmer_dict.values()))
    # normalising the array by dividing every value with the highest count value
    k_array = k_array / np.amax(k_array)

    return k_array

# Generating the input and labels for train, validation and test sets
# The encoding methods for input (x) defined earlier are now applied to the sequences, and the labels (y) are one-hot-encoded using the to_categorical function in keras_utils, thereby, converting the data into the correct format for feeding it to the deep learning models.
# Every array is saved to reduce memory requirements.

# Encoding the sequences into k-mer counts and one-hot-encoded sequences

# ENCODING SEQUENCES for the CNN model in 2 processing variations (4-mer and 7-mer)
# FOR CNN  |  with 4-mer
x_train_CNN_na4 = np.array(train_na['Sequence'].apply(lambda x: one_hot_k(x, four_mers)).tolist())
np.save('arrays/CNN/x_train_CNN_na4', x_train_CNN_na4)

x_train_CNN_a4 = np.array(train_a['Sequence'].apply(lambda x: one_hot_k(x, four_mers)).tolist())
np.save('arrays/CNN/x_train_CNN_a4', x_train_CNN_a4)

x_test_CNN_na4 = np.array(test_na['Sequence'].apply(lambda x: one_hot_k(x, four_mers)).tolist())
np.save('arrays/CNN/x_test_CNN_na4', x_test_CNN_na4)

dataval_CNN_na4 = np.array(val_na['Sequence'].apply(lambda x: one_hot_k(x, four_mers)).tolist())
np.save('arrays/CNN/dataval_CNN_na4', dataval_CNN_na4)

dataval_CNN_a4 = np.array(val_a['Sequence'].apply(lambda x: one_hot_k(x, four_mers)).tolist())
np.save('arrays/CNN/dataval_CNN_a4', dataval_CNN_a4)
print('4-mer complete')
# --------------------------------------------------------------------------------------------------------------------
# FOR CNN  |  with 7-mer
x_train_CNN_na7 = np.array(train_na['Sequence'].apply(lambda x: one_hot_k(x, seven_mers)).tolist())
np.save('arrays/CNN/x_train_CNN_na7', x_train_CNN_na7)

x_train_CNN_a7 = np.array(train_a['Sequence'].apply(lambda x: one_hot_k(x, seven_mers)).tolist())
np.save('arrays/CNN/x_train_CNN_a7', x_train_CNN_a7)

x_test_CNN_na7 = np.array(test_na['Sequence'].apply(lambda x: one_hot_k(x, seven_mers)).tolist())
np.save('arrays/CNN/x_test_CNN_na7', x_test_CNN_na7)

dataval_CNN_na7 = np.array(val_na['Sequence'].apply(lambda x: one_hot_k(x, seven_mers)).tolist())
np.save('arrays/CNN/dataval_CNN_na7', dataval_CNN_na7)

dataval_CNN_a7 = np.array(val_a['Sequence'].apply(lambda x: one_hot_k(x, seven_mers)).tolist())
np.save('arrays/CNN/dataval_CNN_a7', dataval_CNN_a7)
print('7-mer complete')
# --------------------------------------------------------------------------------------------------------------------
print('CNN sequences complete')
