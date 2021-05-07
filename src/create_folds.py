import pandas as pd
from sklearn import model_selection 

import config

if __name__ == '__main__':
    df = pd.read_csv(config.TRAIN_DATA_PATH)
    df['kfold'] = -1
    id_len = df.shape[0]
    df['id'] = list(range(id_len))

    df = df.sample(frac=1).reset_index(drop=True)

    skf = model_selection.StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=df,y=df.target_class.values)):
        df.loc[val_idx,'kfold'] = fold

    print(df['kfold'].value_counts())
    print(f"shape={df.shape}")
    df.to_csv(config.TRAIN_FOLDS,index=False)