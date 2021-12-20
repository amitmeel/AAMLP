import pandas as pd
from sklearn import model_selection

import config

def create_folds(df, n_folds):
    """
    Creates n_folds from the dataset and save it in a csv file
    """
    # create a new column called kfold and fill it with -1
    df['kfold'] = -1
    #randomize the raw of data
    df = df.sample(frac=1).reset_index(drop=True)

    folds = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold_n, (train_idx, val_idx) in enumerate(folds.split(X=df, y=df.label)):
        print("Fold {}".format(fold_n))
        print("Train indices: {}".format(train_idx))
        print("Val indices: {}".format(val_idx))
        df.loc[val_idx, 'kfold'] = fold_n
        
    return df

if __name__=='__main__':
    df = pd.read_csv(config.train_file_path)
    df = create_folds(df, config.n_folds)
    print(df.shape)
    df.to_csv(config.train_fold_file_path, index=False)