# LOADING PACKAGES
import numpy as np
import pandas as pd
INPUT_SHAPE_3_MER, INPUT_SHAPE_5_MER, INPUT_SHAPE_7_MER = (5**3, 1), (5**5, 1), (5**7, 1)   # n**k
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
alphabet_noN = "AGTC"
tri_mers = [''.join(chars) for chars in product(*(3*(alphabet,)))]
fiv_mers = [''.join(chars) for chars in product(*(5*(alphabet,)))]
sev_mers = [''.join(chars) for chars in product(*(7*(alphabet,)))]
sev_mers_no_N = [''.join(chars) for chars in product(*(7*(alphabet,)))]

def one_hot_k(sequence, kmers):
    k = len(kmers[0])
    # define a counter dictionary
    kmer_dict = dict.fromkeys(kmers, 0)

    # standardize the sequence
    # by replacing U with T and all ambiguous bases with N
    sequence = sequence.replace('U', 'T').replace('Y', 'N').replace('R', 'N').replace('W', 'N').replace('S', 'N').replace('K', 'N').replace('M', 'N').replace('D', 'N').replace('V', 'N').replace('H', 'N').replace('B', 'N').replace('X', 'N').replace('-', 'N')
    # count every k-mer in the sequence
    for i in range(0, len(sequence) - k+1):
        kmer_dict[sequence[i:i+k]] += 1

    # k-mer frequency array from the dictionary values
    k_array = np.array(list(kmer_dict.values()))
    # normalizing the array by dividing every value with the highest count value
    k_array = k_array / np.amax(k_array)
    return k_array

####################################################################################################

# ENCODING SEQUENCES for the CNN models in 3 processing variations
# FOR CNN  |  with 3-mer encoding
x_train_CNN_na3 = np.array(
    train_na['Sequence'].apply(lambda x: one_hot_k(x, tri_mers)).tolist())
np.save('arrays/CNN/x_train_CNN_na3.npy', x_train_CNN_na3)

# x_train_CNN_a3 = np.array(
#   train_a['Sequence'].apply(lambda x: one_hot_k(x, tri_mers)).tolist())
# np.save('arrays/CNN/x_train_CNN_a3.npy', x_train_CNN_a3)

x_test_CNN_na3 = np.array(
    test_na['Sequence'].apply(lambda x: one_hot_k(x, tri_mers)).tolist())
np.save('arrays/CNN/x_test_CNN_na3.npy', x_test_CNN_na3)

dataval_CNN_na3 = np.array(
    val_na['Sequence'].apply(lambda x: one_hot_k(x, tri_mers)).tolist())
np.save('arrays/CNN/dataval_CNN_na3.npy', dataval_CNN_na3)

# dataval_CNN_a3 = np.array(
#   val_a['Sequence'].apply(lambda x: one_hot_k(x, tri_mers)).tolist())
# np.save('arrays/CNN/dataval_CNN_a3.npy', dataval_CNN_a3)
print('3-mer encoding complete')
# -----------------------------------------------------------------------------------------------------
# FOR CNN  |  with 5-mer encoding
x_train_CNN_na5 = np.array(
    train_na['Sequence'].apply(lambda x: one_hot_k(x, fiv_mers)).tolist())
np.save('arrays/CNN/x_train_CNN_na5.npy', x_train_CNN_na5)

# x_train_CNN_a5 = np.array(
#   train_a['Sequence'].apply(lambda x: one_hot_k(x, fiv_mers)).tolist())
# np.save('arrays/CNN/x_train_CNN_a5.npy', x_train_CNN_a5)

x_test_CNN_na5 = np.array(
    test_na['Sequence'].apply(lambda x: one_hot_k(x, fiv_mers)).tolist())
np.save('arrays/CNN/x_test_CNN_na5.npy', x_test_CNN_na5)

dataval_CNN_na5 = np.array(
    val_na['Sequence'].apply(lambda x: one_hot_k(x, fiv_mers)).tolist())
np.save('arrays/CNN/dataval_CNN_na5.npy', dataval_CNN_na5)

# dataval_CNN_a5 = np.array(
#   val_a['Sequence'].apply(lambda x: one_hot_k(x, fiv_mers)).tolist())
# np.save('arrays/CNN/dataval_CNN_a5.npy', dataval_CNN_a5)
print('5-mer encoding complete')
# -----------------------------------------------------------------------------------------------------
# FOR CNN  |  with 7-mer encoding
x_train_CNN_na7 = np.array(
    train_na['Sequence'].apply(lambda x: one_hot_k(x, sev_mers)).tolist())
np.save('arrays/CNN/x_train_CNN_na7.npy', x_train_CNN_na7)

x_train_CNN_a7 = np.array(
    train_a['Sequence'].apply(lambda x: one_hot_k(x, sev_mers)).tolist())
np.save('arrays/CNN/x_train_CNN_a7.npy', x_train_CNN_a7)

x_test_CNN_na7 = np.array(
    test_na['Sequence'].apply(lambda x: one_hot_k(x, sev_mers)).tolist())
np.save('arrays/CNN/x_test_CNN_na7.npy', x_test_CNN_na7)

dataval_CNN_na7 = np.array(
    val_na['Sequence'].apply(lambda x: one_hot_k(x, sev_mers)).tolist())
np.save('arrays/CNN/dataval_CNN_na7.npy', dataval_CNN_na7)

dataval_CNN_a7 = np.array(
    val_a['Sequence'].apply(lambda x: one_hot_k(x, sev_mers)).tolist())
np.save('arrays/CNN/dataval_CNN_a7.npy', dataval_CNN_a7)
print('7-mer encoding complete')
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# FOR CNN  |  with 7-mer encoding ignoring N
x_train_CNN_na7N = np.array(
    train_na['Sequence'].apply(lambda x: one_hot_k(x, sev_mers_no_N)).tolist())
np.save('arrays/CNN/x_train_CNN_na7N.npy', x_train_CNN_na7N)

# x_train_CNN_a7N = np.array(
#     train_a['Sequence'].apply(lambda x: one_hot_k(x, sev_mers_no_N)).tolist())
# np.save('arrays/CNN/x_train_CNN_a7N.npy', x_train_CNN_a7N)

x_test_CNN_na7N = np.array(
    test_na['Sequence'].apply(lambda x: one_hot_k(x, sev_mers_no_N)).tolist())
np.save('arrays/CNN/x_test_CNN_na7N.npy', x_test_CNN_na7N)

dataval_CNN_na7N = np.array(
    val_na['Sequence'].apply(lambda x: one_hot_k(x, sev_mers_no_N)).tolist())
np.save('arrays/CNN/dataval_CNN_na7N.npy', dataval_CNN_na7N)

# dataval_CNN_a7N = np.array(
#     val_a['Sequence'].apply(lambda x: one_hot_k(x, sev_mers_no_N)).tolist())
# np.save('arrays/CNN/dataval_CNN_a7N.npy', dataval_CNN_a7N)
print('7-mer encoding, ignoring N complete')
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# FOR CNN  |  with 7-mer encoding on V-region selected sequences (CURRENTLY NOT USED)
# x_train_CNN_na7V = np.array(
#     train_na['V_Sequence'].apply(lambda x: one_hot_k(x, sev_mers)).tolist())
# np.save('arrays/CNN/x_train_CNN_na7V.npy', x_train_CNN_na7V)

# x_train_CNN_a7V = np.array(
#     train_a['V_Sequence'].apply(lambda x: one_hot_k(x, sev_mers)).tolist())
# np.save('arrays/CNN/x_train_CNN_a7V.npy', x_train_CNN_a7V)

# x_test_CNN_na7V = np.array(
#     test_na['V_Sequence'].apply(lambda x: one_hot_k(x, sev_mers)).tolist())
# np.save('arrays/CNN/x_test_CNN_na7V.npy', x_test_CNN_na7V)

# dataval_CNN_na7V = np.array(
#     val_na['V_Sequence'].apply(lambda x: one_hot_k(x, sev_mers)).tolist())
# np.save('arrays/CNN/dataval_CNN_na7V.npy', dataval_CNN_na7V)

# dataval_CNN_a7V = np.array(
#     val_a['V_Sequence'].apply(lambda x: one_hot_k(x, sev_mers)).tolist())
# np.save('arrays/CNN/dataval_CNN_a7V.npy', dataval_CNN_a7V)
# print('7-mer encoding for V-regions complete')
# -----------------------------------------------------------------------------------------------------
print('CNN sequences complete')

####################################################################################################