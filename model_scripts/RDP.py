# LOADING PACKAGES
import pandas as pd
import sys
import time
import os
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
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

global RDPfiles
import pandas as pd
# runs to do
#   - genus level with 3 data types (non-augmented, augmented, V-selected)
#   - family and species level with non-augmented

# variables
RDPfiles = "RDPfiles"
classifier_loc = "rdptools/classifier.jar"
confidence_score = 0.8
level = 'genus'             # used for evaluation
model_run = 'RDP_gen'       # used for naming the run (change taxon)
data = 'x_train_RDP_naX'    # used for naming the data (change na/n/V)
seq_col = -1                # used for selecting sequence (change -1/-2 for V_seq col)
is_v = -1                   # used for selecting taxa (change to -2 if there is a V_seq col)
# loading data
train_data = pd.read.csv('df_train_0.csv')  # change 1/0 for a/na data
val_data = pd.read_csv('df_val_0.csv')      # change 1/0 for a/na data
test_data = pd.read_csv('df_test_0.csv')

print('Variables set and datasets loaded!')

####################################################################################################

def lineage2taxTrain(raw_taxons):
    taxons_list = raw_taxons.strip().split('\n')
    header = taxons_list[0].split('\t')[1:] # headers = list of ranks
    hash = {} # taxon name-id map
    ranks = {} # column number-rank map
    lineages = [] # list of unique lineages

    with open("{}/ready4train_taxonomy.txt".format(RDPfiles), "w") as f:
        # initiate root rank taxon id map
        hash = {"Root":0}
        for i in range(len(header)):
            name = header[i]
            ranks[i] = name

        # root rank info
        root = ['0', 'Root', '-1', '0', 'rootrank']
        f.write("*".join(root) +  '\n')

        ID = 0
        for line in taxons_list[1:]:
            cols = line.strip().split('\t')[1:]
            # iterate each column
            for i in range(len(cols)):
                name = []
                for node in cols[:i + 1]:
                    node = node.strip()
                    if not node in ('-', ''):
                        name.append(node)

                pName = ";".join(name[:-1])
                if not name in lineages:
                    lineages.append(name)

                depth = len(name)
                name = ';'.join(name)
                if name in hash.keys():
                    # already seen this lineage
                    continue
                try:
                    rank = ranks[i]
                except KeyError:
                    print (cols)
                    sys.exit()

                if i == 0:
                    pName = 'Root'
                # parent taxid
                pID = hash[pName]
                ID += 1
                # add name-id to the map
                hash[name] = ID
                out = ['%s'%ID, name.split(';')[-1], '%s'%pID, '%s'%depth, rank]
                f.write("*".join(out) + '\n')
    f.close()

def addFullLineage(raw_taxons, raw_seqs):
    # lineage map
    hash = {}
    taxonomy_list = raw_taxons.strip().split('\n')

    for line in taxonomy_list[1:]:
        line = line.strip()
        cols = line.strip().split('\t')
        lineage = ['Root']

        for node in cols[1:]:
            node = node.strip()
            if not (node == '-' or node == ''):
                lineage.append(node)

        ID = cols[0]
        lineage = ';'.join(lineage).strip()
        hash[ID] = lineage

    sequence_list = raw_seqs.strip().split('\n')
    with open("{}/ready4train_seqs.fasta".format(RDPfiles), "w") as f:
        for line in sequence_list:
            line = line.strip()
            if line == '':
                continue
            if line[0] == '>':
                ID = line.strip().split()[0].replace('>', '')
                lineage = hash[ID]
                f.write('>' + ID + '\t' + lineage + '\n')
            else:
                f.write(line.strip() + '\n')
    f.close()

def RDPoutput2score(pred_file, true_file, level, cf):
    taxon_list = []
    ranks = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    level = ranks.index(level)

    pred = pd.read_csv(pred_file, sep="\t", header=None)
    pred.drop(pred.columns[1:level+5+2*level],  axis = 'columns', inplace=True)
    pred.drop(pred.columns[4:], axis = 'columns', inplace=True)
    
    pred_dict = {}
    for index, row in pred.iterrows():
        row = row.tolist()
        if row[1] not in taxon_list:
            taxon_list += [row[1]]
        if float(row[3]) >= cf:
            pred_dict[row[0]] = row[1]

    true = pd.read_csv(true_file, sep="\t", header=None)
    true_dict = {}
    for index, row in true.iterrows():
        true_dict[row[0]] = row[level+1]
        if row[level+1] not in taxon_list:
            taxon_list += [row[level+1]]


    y_pred, y_true = [], []
    for i in pred_dict.keys():
        y_pred.append(taxon_list.index(pred_dict[i]))
        y_true.append(taxon_list.index(true_dict[i]))

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)

    score_dict = pd.DataFrame({
        'Model/run' : model_run,    # 'RDP_gen'
        'Data' : data,              # 'x_train_RDP_naX'
        'Training time' : None, 
        'Test loss' : None, 
        'Test accuracy' : acc, 
        'F1-score' : f1, 
        'MCC' : mcc}, 
        index = [0]
        )
    print(score_dict)
    return score_dict

####################################################################################################

# main_RDP.py
os.system("mkdir {}".format(RDPfiles))

# merge train and validation dataframes into one.
train = pd.concat([train_data, val_data], ignore_index = True)
test = test_data

# convert train and test dataframe into tab separated taxonomy and sequence string
# taxnomy file is converted to a tab separated string
# sequence file is converted to fasta format with sequence ID and sequence
raw_seqs = ''
raw_taxons = 'SeqId Kingdom	Phylum	Class	Order	Family	Genus	Species' + '\n'
for index, row in train.iterrows():
    taxons = row.tolist()
    raw_seqs += '>' + taxons[0] + '\n' + taxons[seq_col] + '\n'
    raw_taxons += '\t'.join(taxons[:is_v]) + '\n'

# convert test dataframe into text and fasta files to be utilized by RDP
# taxnomy file is converted to a tab separated text file
# sequence file is converted to fasta format
with open("{}/test_sequences.fasta".format(RDPfiles), "w") as seq_f, open(
    "{}/test_taxonomy.txt".format(RDPfiles), "w") as tax_f:

    for index, row in test.iterrows():
        taxons = row.tolist()
        seq_f.write('>' + taxons[0] + '\n' + taxons[seq_col] + '\n')
        tax_f.write('\t'.join(taxons[:is_v]) + '\n')
        
    seq_f.close()
    tax_f.close()

# convert raw taxonomy and sequence files to ready4rdp trainable files
lineage2taxTrain(raw_taxons)
addFullLineage(raw_taxons, raw_seqs)

print("Data preprocessing for RDP completed")

####################################################################################################

# Training the RDP classifier
start_time = time.time()
os.system("java -Xmx10g -jar {} train -o {}/training_files -s {}/ready4train_seqs.fasta -t {}/ready4train_taxonomy.txt".format(
    classifier_loc, RDPfiles, RDPfiles, RDPfiles))

with open("{}/training_files/rRNAClassifier.properties".format(RDPfiles), "w") as f:
    f.write("bergeyTree=bergeyTrainingTree.xml\nprobabilityList=genus_wordConditionalProbList.txt\nprobabilityIndex=wordConditionalProbIndexArr.txt\nwordPrior=logWordPrior.txt\nclassifierVersion=RDP Naive Bayesian rRNA Classifier Version 2.5, May 2012 ")
    f.close()

time_taken = round(time.time() - start_time)
print("RDP training-time: {} seconds".format(time_taken))

# Testing the RDP classifier
os.system("java -Xmx10g -jar {} classify -t {}/training_files/rRNAClassifier.properties  -o {}/output.txt {}/test_sequences.fasta".format(
    classifier_loc, RDPfiles, RDPfiles, RDPfiles))

# Evaluating the RDP classifier and save the results
score_dict = RDPoutput2score("{}/output.txt".format(RDPfiles), 
"{}/test_taxonomy.txt".format(RDPfiles), level, confidence_score)

score_dict.at[0, 'Training time'] = time_taken
score_dict.to_csv('scores/RDP_evaluation', index = False)

####################################################################################################

print('RDP training complete!')