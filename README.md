# 16S-ML-models
**Machine learning models for 16S rRNA sequence classification**

This repository contains the code and comparative analyses of 5 machine learning models on different classification tasks and using various preproccessing methods. A list of models used for bacterial taxonomy classification with the curated 16S rRNA gene is as follows:
- **Ribosomal Database Project (RDP) Classifier** with k-mer frequency classification

    This model was developed by Wang, Q. et al (2007).
    Access the [github repository](https://github.com/rdpstaff/classifier 'RDP Classifier code') and the [paper](https://doi.org/10.1128/AEM.00062-07 'RDP Classifier paper')
    
- **Convolutional Neural Networks (CNN)** with k-mer frequency classification

    This model is based on an architecture developed by Fiannaca, A. et al (2018).
    Access the [github repository](https://github.com/IcarPA-TBlab/MetagenomicDC) and the [paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2182-6)
    
- **Bilateral Long-Short Term Memory NN (BiLSTM)** with one-hot-encoded sequence classification
    
    This model is based on an architecture developed by Philipp Münch.
    Access the [github repository](https://github.com/philippmuench/dna_lstm)
    
- **Combined Convolutional BiLSTM (ConvBiLSTM)** with one-hot-encoded sequence classification
    
    This model is based on an architecture developed by Desai, P. et al (2020). 
    Access the [paper](https://doi.org/10.1007/978-3-030-57821-3_25 'ConvBiLSTM paper')
    
- **Attention-based ConvBiLSTM (Read2Pheno)** with one-hot-encoded sequence classification
    
    This model is based on an architecture developed by Zhao, Z. et al (2021). 
    Access the [github repository](https://github.com/z2e2/seq2att 'Read2Pheno code') and the [paper](https://doi.org/10.1371/journal.pcbi.1009345 'Read2Pheno paper')


These models have been combined in the jupyter notebook file (models_notebook.ipynb). This notebook also contains the scripts required for preprocessing the data and labels, compiling and running the models, and saving and visualising the results.

The seperate data-processing and model-training scripts can be use when the memory requirements for running the complete jupyter notebook are too high for the user's system.
