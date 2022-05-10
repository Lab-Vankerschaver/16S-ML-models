# 16S-ML-models
**Machine learning models for 16S rRNA sequence classification**

This repository contains the code and comparative analyses of 5 machine learning models on different classification tasks and using various preproccessing methods:
- **Ribosomal Database Project (RDP) Classifier** with k-mer frequency classification

    This model was developed by Wang Q. et al (2007).
    Access the [code](https://github.com/rdpstaff/classifier 'RDP Classifier code') and the [paper](https://doi.org/10.1128/AEM.00062-07 'RDP Classifier paper')
    
- **Convolutional Neural Network (CNN)** with k-mer frequency classification

    This model...
    
- **Bilateral Long-Short Term Memomry NN (BiLSTM)** with one-hot-encoded sequence classification
    
    This model is based on an architecture developed by Philipp Münch.
    Acess the [github repository](https://github.com/philippmuench/dna_lstm)
    
- **Combined Convolutional BiLSTM (ConvBiLSTM)** with one-hot-encoded sequence classification
    
    This model is based on an architecture developed by Desai P. et al (2020). 
    Access the [paper](https://doi.org/10.1007/978-3-030-57821-3_25 'ConvBiLSTM paper')
    
- **Attention-based ConvBiLSTM (Read2Pheno)** with one-hot-encoded sequence classification
    
    This model is based on an architecture developed by Zhao Z. et al (2021). 
    Access the [code](https://github.com/z2e2/seq2att 'Read2Pheno code') and the [paper](https://doi.org/10.1371/journal.pcbi.1009345 'Read2Pheno paper')


These models have been combined in the jupyter notebook file (models_notebook). This notebook also contains the scripts required for processing the data and labels, compiling and running the models, and saving and visualizing the results.
