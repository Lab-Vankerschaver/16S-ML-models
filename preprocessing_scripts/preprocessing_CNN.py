# LOADING PACKAGES
import numpy as np
import pandas as pd
print('Packages loaded!')

####################################################################################################

# LOADING THE NON-AUGMENTED AND AUGMENTED DATASETS
train_na = pd.read_csv('df_train_0.csv')
val_na = pd.read_csv('df_val_0.csv')
test_na = pd.read_csv('df_test_0.csv')
# ------------------------------------------------
train_a = pd.read_csv('df_train_1.csv')
val_a = pd.read_csv('df_val_1.csv')
# ------------------------------------------------
print('Datasets loaded!')

####################################################################################################

from itertools import product
# define all possible k-mers
alphabet = "AGTCN"
tri_mers = [''.join(chars) for chars in product(*(3*(alphabet,)))]
six_mers = [''.join(chars) for chars in product(*(6*(alphabet,)))]

def one_hot_k(sequence, kmers):
    k = len(kmers[0])
    # define a counter dictionary
    kmer_dict = dict.fromkeys(kmers, 0)

    # standardize the sequence
    # by replacing U with T and all ambiguous bases with N
    sequence = sequence.replace('U', 'T').replace('Y', 'N').replace('R', 'N').replace('W', 'N').replace('S', 'N').replace('K', 'N').replace('M', 'N').replace('D', 'N').replace('V', 'N').replace('H', 'N').replace('B', 'N').replace('X', 'N').replace('-', 'N')
    # count every k-mer in the sequence
    for i in range(0, len(sequence) -k+1):
        kmer_dict[sequence[i:i+k]] += 1

    # k-mer frequency array from the dictionary values
    k_array = np.array(list(kmer_dict.values()))
    # normalizing the array by dividing every value with the highest count value
    k_array = k_array / np.amax(k_array)
    return k_array

####################################################################################################

# ENCODING SEQUENCES for the CNN models in 2 processing variations
# FOR CNN  |  with 3-mer
x_train_CNN_na3 = np.array(train_na['Sequence'].apply(lambda x: one_hot_k(x, tri_mers)).tolist())
np.save('arrays/CNN/x_train_CNN_na3.npy', x_train_CNN_na3)

# x_train_CNN_a3 = np.array(train_a['Sequence'].apply(lambda x: one_hot_k(x, tri_mers)).tolist())
# np.save('arrays/CNN/x_train_CNN_a3.npy', x_train_CNN_a3)

x_test_CNN_na3 = np.array(test_na['Sequence'].apply(lambda x: one_hot_k(x, tri_mers)).tolist())
np.save('arrays/CNN/x_test_CNN_na3.npy', x_test_CNN_na3)

dataval_CNN_na3 = np.array(val_na['Sequence'].apply(lambda x: one_hot_k(x, tri_mers)).tolist())
np.save('arrays/CNN/dataval_CNN_na3.npy', dataval_CNN_na3)

# dataval_CNN_a3 = np.array(val_a['Sequence'].apply(lambda x: one_hot_k(x, tri_mers)).tolist())
# np.save('arrays/CNN/dataval_CNN_a3.npy', dataval_CNN_a3)
print('3-mer complete')
# -----------------------------------------------------------------------------------------------------
# FOR CNN  |  with 6-mer
x_train_CNN_na6 = np.array(train_na['Sequence'].apply(lambda x: one_hot_k(x, six_mers)).tolist())
np.save('arrays/CNN/x_train_CNN_na6.npy', x_train_CNN_na6)

x_train_CNN_a6 = np.array(train_a['Sequence'].apply(lambda x: one_hot_k(x, six_mers)).tolist())
np.save('arrays/CNN/x_train_CNN_a6.npy', x_train_CNN_a6)

x_test_CNN_na6 = np.array(test_na['Sequence'].apply(lambda x: one_hot_k(x, six_mers)).tolist())
np.save('arrays/CNN/x_test_CNN_na6.npy', x_test_CNN_na6)

dataval_CNN_na6 = np.array(val_na['Sequence'].apply(lambda x: one_hot_k(x, six_mers)).tolist())
np.save('arrays/CNN/dataval_CNN_na6.npy', dataval_CNN_na6)

dataval_CNN_a6 = np.array(val_a['Sequence'].apply(lambda x: one_hot_k(x, six_mers)).tolist())
np.save('arrays/CNN/dataval_CNN_a6.npy', dataval_CNN_a6)
print('6-mer complete')
# -----------------------------------------------------------------------------------------------------
print('CNN sequences complete')

####################################################################################################