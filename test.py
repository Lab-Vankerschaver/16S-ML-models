def get_taxon_dict(df, taxon):
    # listing all unique taxon labels
    taxon_list = list(df[taxon].unique())
    # generating a dictionary to associate every unique taxon to a number
    return dict(zip(taxon_list, range(0, len(taxon_list))))
  
y_train_gen_na = to_categorical(
    y = train_na[taxon].map(taxon_dict).astype(np.float32))
