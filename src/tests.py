import pandas as pd
import numpy as np
import config

if __name__ == '__main__':
    df = pd.read_csv(config.TRAIN_FOLDS)
    print(df['target_class'].head(20).values,'\n',df['target_class'].rank().head(20).values)
