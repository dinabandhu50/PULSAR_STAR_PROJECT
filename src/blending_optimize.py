import glob 
import pandas as pd
from sklearn import metrics
from scipy import fmin

def objective():
    pass

if __name__ == '__main__':
    files = glob.glob('..\\model_preds\\*.csv')
    df = None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = df.merge(temp_df,on="id",how="left")

    pred_cols = ['lr_pred','rf_pred']
    # get optimal weights
    
    # blending with weighted average method
    w0, w1 = 2, 1
    weighted_average_pred = w0 * df['lr_pred'].values + w1 * df['rf_pred'].values
    print('blending result: ',metrics.roc_auc_score(df['target_class_x'].values,weighted_average_pred))

