# LOADING PACKAGES
import numpy as np
import pandas as pd
MAX_LEN, MAX_LEN_V, INPUT_SHAPE_RNN, INPUT_SHAPE_RNN_V, = 2000, 500, (2000, 4), (500, 4)    # cut-off length
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

# ### One-hot-encoding the sequences
# For use in the various Recurrent Neural Networks (RNN), the nucleotide sequences are processed into a one-hot-encoded format.

# Dictionary without consideration of mutation rate
one_hot_dict0 = {
    'A': [1.,0.,0.,0.], 'G':[0.,1.,0.,0.], 'T':[0.,0.,1.,0.], 'U':[0.,0.,1.,0.], 'C':[0.,0.,0.,1.], 
    'Y':[0.,0.,0.5,0.5], 'R':[0.5,0.5,0.,0.], 'W':[0.5,0.,0.5,0.], 'S':[0.,0.5,0.,0.5], 'K':[0.,0.5,0.5,0.], 'M':[0.5,0.,0.,0.5], 
    'D':[0.33,0.33,0.33,0.], 'V':[0.33,0.33,0.,0.33], 'H':[0.33,0.,0.33,0.33], 'B':[0.,0.33,0.33,0.33], 
    'X':[0.25,0.25,0.25,0.25], 'N':[0.25,0.25,0.25,0.25], '-':[0.,0.,0.,0.]
    }
# Dictionary with consideration of mutation rate (transition >> transversion)
one_hot_dict1 = {
    'A': [1.,0.,-0.5,-0.5], 'G':[0.,1.,-0.5,-0.5], 'T':[-0.5,-0.5,1.,0.], 'U':[-0.5,-0.5,1.,0.], 'C':[-0.5,-0.5,0.,1.], 
    'Y':[-0.5,-0.5,0.5,0.5], 'R':[0.5,0.5,-0.5,-0.5], 'W':[0.5,-0.5,0.5,-0.5], 'S':[-0.5,0.5,-0.5,0.5], 'K':[-0.5,0.5,0.5,-0.5], 'M':[0.5,-0.5,-0.5,0.5], 
    'D':[0.33,0.33,0.33,-1.], 'V':[0.33,0.33,-1.,0.33], 'H':[0.33,-1.,.33,0.33], 'B':[-1.,0.33,0.33,0.33], 
    'X':[0.,0.,0.,0.], 'N':[0.,0.,0.,0.], '-':[0.,0.,0.,0.]
    }

def one_hot_seq(sequence, one_hot_dict, max_len):
    # padding the sequences to a fixed length
	sequence += '-'*(max_len - len(sequence))
    # generating list of one-hot-lists using the dictionary
	onehot_encoded = [one_hot_dict[nucleotide] for nucleotide in sequence]
    # returning the list of lists as a numpy array
	return np.array(onehot_encoded)

####################################################################################################

# ENCODING SEQUENCES for the RNN models in 2 processing variations
# FOR RNN  |  with regular one-hot-encoding
x_train_RNN_na0 = np.array(
    train_na['Sequence'].apply(lambda x: one_hot_seq(x, one_hot_dict0, max_len = MAX_LEN)).tolist())
np.save('arrays/RNN/x_train_RNN_na0.npy', x_train_RNN_na0)

# x_train_RNN_a0 = np.array(
#   train_a['Sequence'].apply(lambda x: one_hot_seq(x, one_hot_dict0, max_len = MAX_LEN)).tolist())
# np.save('arrays/RNN/x_train_RNN_a0.npy', x_train_RNN_a0)

x_test_RNN_na0 = np.array(
    test_na['Sequence'].apply(lambda x: one_hot_seq(x, one_hot_dict0, max_len = MAX_LEN)).tolist())
np.save('arrays/RNN/x_test_RNN_na0.npy', x_test_RNN_na0)

dataval_RNN_na0 = np.array(
    val_na['Sequence'].apply(lambda x: one_hot_seq(x, one_hot_dict0, max_len = MAX_LEN)).tolist())
np.save('arrays/RNN/dataval_RNN_na0.npy', dataval_RNN_na0)

# dataval_RNN_a0 = np.array(
#   val_a['Sequence'].apply(lambda x: one_hot_seq(x, one_hot_dict0, max_len = MAX_LEN)).tolist())
# np.save('arrays/RNN/dataval_RNN_a0.npy', dataval_RNN_a0)
print('Regular one-hot-encoding complete')
# -----------------------------------------------------------------------------------------------------
# FOR RNN  |  with matation rate adjusted one-hot-encoding
x_train_RNN_na1 = np.array(
    train_na['Sequence'].apply(lambda x: one_hot_seq(x, one_hot_dict1, max_len = MAX_LEN)).tolist())
np.save('arrays/RNN/x_train_RNN_na1.npy', x_train_RNN_na1)

x_train_RNN_a1 = np.array(
    train_a['Sequence'].apply(lambda x: one_hot_seq(x, one_hot_dict1, max_len = MAX_LEN)).tolist())
np.save('arrays/RNN/x_train_RNN_a1.npy', x_train_RNN_a1)

x_test_RNN_na1 = np.array(
    test_na['Sequence'].apply(lambda x: one_hot_seq(x, one_hot_dict1, max_len = MAX_LEN)).tolist())
np.save('arrays/RNN/x_test_RNN_na1.npy', x_test_RNN_na1)

dataval_RNN_na1 = np.array(
    val_na['Sequence'].apply(lambda x: one_hot_seq(x, one_hot_dict1, max_len = MAX_LEN)).tolist())
np.save('arrays/RNN/dataval_RNN_na1.npy', dataval_RNN_na1)

dataval_RNN_a1 = np.array(
    val_a['Sequence'].apply(lambda x: one_hot_seq(x, one_hot_dict1, max_len = MAX_LEN)).tolist())
np.save('arrays/RNN/dataval_RNN_a1.npy', dataval_RNN_a1)
print('Mutations rate adjusted one-hot-encoding complete')
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# FOR RNN  |  with matation rate adjusted one-hot-encoding on the V-region selected sequences (CURRENTLY NOT USED)
# x_train_RNN_na1V = np.array(
#     train_na['V_Sequence'].apply(lambda x: one_hot_seq(x, one_hot_dict1, max_len = MAX_LEN_V)).tolist())
# np.save('arrays/RNN/x_train_RNN_na1V.npy', x_train_RNN_na1V)

# x_train_RNN_a1V = np.array(
#   train_a['V_Sequence'].apply(lambda x: one_hot_seq(x, one_hot_dict1, max_len = MAX_LEN_V)).tolist())
# np.save('arrays/RNN/x_train_RNN_a1V.npy', x_train_RNN_a1V)

# x_test_RNN_na1V = np.array(
#     test_na['V_Sequence'].apply(lambda x: one_hot_seq(x, one_hot_dict1, max_len = MAX_LEN_V)).tolist())
# np.save('arrays/RNN/x_test_RNN_na1V.npy', x_test_RNN_na1V)

# dataval_RNN_na1V = np.array(
#     val_na['V_Sequence'].apply(lambda x: one_hot_seq(x, one_hot_dict1, max_len = MAX_LEN_V)).tolist())
# np.save('arrays/RNN/dataval_RNN_na1V', dataval_RNN_na1V)

# dataval_RNN_a1V = np.array(
#   val_a['V_Sequence'].apply(lambda x: one_hot_seq(x, one_hot_dict1, max_len = MAX_LEN_V)).tolist())
# np.save('arrays/RNN/dataval_RNN_a1V.npy', dataval_RNN_a1V)
# print('Mutations rate adjusted one-hot-encoding V-region complete')
# -----------------------------------------------------------------------------------------------------
print('RNN sequences complete')

####################################################################################################