import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

# LOADING THE NON-AUGMENTED AND AUGMENTED DATASETS

train_na = pd.read_csv('df_train_0.csv')
val_na = pd.read_csv('df_val_0.csv')
test_na = pd.read_csv('df_test_0.csv')

train_a = pd.read_csv('df_train_1.csv')
val_a = pd.read_csv('df_val_1.csv')

# One-hot-encoding the labels
# For use in the deep learning models, the labels are processed into a one-hot-encoding format. To achieve this, every unique label is first encoded to a numerical value.

def get_taxon_dict(df, taxon):
    # listing all unique taxon labels
    taxon_list = list(df[taxon].unique())

    # generating a dictionary to associate every unique taxon to a number
    taxon_dict = dict(zip(taxon_list, range(0, len(taxon_list))))
    # and the reversed dictionary as a lookup table
    taxon_dict_lookup = {v: k for k, v in taxon_dict.items()}

    return taxon_dict, taxon_dict_lookup


# Generating one-hot encoded train, validation and test data at the Family level
#---------------------------------------------------------------------------------------------------------------------
# ONE-HOT-ENCODING LABELS
taxon = 'Family'
taxon_dict = get_taxon_dict(test_na, taxon)[0]

# Associate every entry's label in the df to a number using the dictionary & one-hot encode the numerical labels
y_train_fam_na = to_categorical(y=train_na[taxon].map(taxon_dict).astype(np.float32))
np.save('arrays/family/y_train_fam_na.npy', y_train_fam_na)

y_train_fam_a = to_categorical(y=train_a[taxon].map(taxon_dict).astype(np.float32))
np.save('arrays/family/y_train_fam_a.npy', y_train_fam_a)

y_test_fam_na = to_categorical(y=test_na[taxon].map(taxon_dict).astype(np.float32))
np.save('arrays/family/y_test_fam_na.npy', y_test_fam_na)

labelsval_fam_na = to_categorical(y=val_na[taxon].map(taxon_dict).astype(np.float32))
np.save('arrays/family/labelsval_fam_na.npy', labelsval_fam_na)

labelsval_fam_a = to_categorical(y=val_a[taxon].map(taxon_dict).astype(np.float32))
np.save('arrays/family/labelsval_fam_a.npy', labelsval_fam_a)
print('Family label arrays generated')
#---------------------------------------------------------------------------------------------------------------------
fam_count = train_na[taxon].nunique()
print(f'The number of unique family labels: {fam_count}')

# ### At genus level

# Generating one-hot encoded train, validation and test data at the Genus level
#---------------------------------------------------------------------------------------------------------------------
# ONE-HOT-ENCODING LABELS
taxon = 'Genus'
taxon_dict = get_taxon_dict(test_na, taxon)[0]

y_train_gen_na = to_categorical(y=train_na[taxon].map(taxon_dict).astype(np.float32))
np.save('arrays/genus/y_train_gen_na.npy', y_train_gen_na)

y_train_gen_a = to_categorical(y=train_a[taxon].map(taxon_dict).astype(np.float32))
np.save('arrays/genus/y_train_gen_a.npy', y_train_gen_a)

y_test_gen_na = to_categorical(y=test_na[taxon].map(taxon_dict).astype(np.float32))
np.save('arrays/genus/y_test_gen_na.npy', y_test_gen_na)

labelsval_gen_na = to_categorical(y=val_na[taxon].map(taxon_dict).astype(np.float32))
np.save('arrays/genus/labelsval_gen_na.npy', labelsval_gen_na)

labelsval_gen_a = to_categorical(y=val_a[taxon].map(taxon_dict).astype(np.float32))
np.save('arrays/genus/labelsval_gen_a.npy', labelsval_gen_a)
print('Genus label arrays generated')
#---------------------------------------------------------------------------------------------------------------------
gen_count = train_na[taxon].nunique()
print(f'The number of unique genus labels: {gen_count}')

# ### At species level

# Generating one-hot encoded train, validation and test data at the Species level
#---------------------------------------------------------------------------------------------------------------------
# ONE-HOT-ENCODING LABELS
taxon = 'Species'
taxon_dict = get_taxon_dict(test_na, taxon)[0]

y_train_spe_na = to_categorical(y=train_na[taxon].map(taxon_dict).astype(np.float32))
np.save('arrays/species/y_train_spe_na.npy', y_train_spe_na)

y_train_spe_a = to_categorical(y=train_a[taxon].map(taxon_dict).astype(np.float32))
np.save('arrays/species/y_train_spe_a.npy', y_train_spe_a)

y_test_spe_na = to_categorical(y=test_na[taxon].map(taxon_dict).astype(np.float32))
np.save('arrays/species/y_test_spe_na.npy', y_test_spe_na)

labelsval_spe_na = to_categorical(y=val_na[taxon].map(taxon_dict).astype(np.float32))
np.save('arrays/species/labelsval_spe_na.npy', labelsval_spe_na)

labelsval_spe_a = to_categorical(y=val_a[taxon].map(taxon_dict).astype(np.float32))
np.save('arrays/species/labelsval_spe_a.npy', labelsval_spe_a)
print('Species label arrays generated')
#---------------------------------------------------------------------------------------------------------------------
spe_count = train_na[taxon].nunique)
print(f'The number of unique species labels: {spe_count}')
