# 16S-ML-models
**Machine learning models for 16S rRNA sequence classification**

The repository contains code and comparative analyses of 5 machine learning models on different classification tasks and using various preproccessing methods:
- **Ribosomal Database Project (RDP) Classifier** with k-mer frequency classification

    This model was developed by Wang Q. et al (2007).
    Their code can be accessed here: github.com/rdpstaff/classifier and the paper here: doi.org/10.1128/AEM.00062-07
    
- **Convolutional Neural Network (CNN)** with k-mer frequency classification

    This model...
    
- **Bilateral Long-Short Term Memomry NN (BiLSTM)** with one-hot-encoded sequence classification
    
    This model...
    
- **Combined Convolutional BiLSTM (ConvBiLSTM)** with one-hot-encoded sequence classification
    
    This model is based on an architecture developed by Desai P. et al (2020). 
    The paper can be accessed here: doi.org/10.1007/978-3-030-57821-3_25
    
- **Attention-based ConvBiLSTM (Read2Pheno)** with one-hot-encoded sequence classification
    
    This model is based on an architecture developed by Zhao Z. et al (2021). 
    Their code can be accessed here: github.com/z2e2/seq2att and the paper here: doi.org/10.1371/journal.pcbi.1009345


These models have been combined in the jupyter notebook file. This notebook also contains the scripts required for processing the data and labels, compiling and running the models, and visualizing the results.
