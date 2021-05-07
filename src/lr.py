import pandas as pd
from sklearn import linear_model
from sklearn import metrics

import config 
from lr_pipeline import lr_pipe

def run_training(fold):
    # load a single fold here
    df = pd.read_csv(config.TRAIN_FOLDS)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_val = df[df.kfold == fold].reset_index(drop=True)

    # get X_train, X_val, y_train and y_val
    X_train = df_train.drop(['id','target_class','kfold'],axis=1)
    y_train = df_train['target_class']

    X_val = df_val.drop(['id','target_class','kfold'],axis=1)
    y_val = df_val['target_class']

    # preprocessing pipeline here
    X_train = lr_pipe.fit_transform(X_train)
    X_val = lr_pipe.transform(X_val)

    # train clf
    clf = linear_model.LogisticRegression()
    clf.fit(X_train,y_train) 

    # metric
    pred = clf.predict_proba(X_val)[:,1]
    auc = metrics.roc_auc_score(y_val,pred)
    print(f"fold={fold}, auc={auc}")

    df_val.loc[:,"lr_pred"] = pred
    return df_val[["id","kfold","target_class","lr_pred"]]

if __name__ == '__main__':
    dfs = []
    for i in range(5):
        temp_df = run_training(i)
        dfs.append(temp_df)
    fin_valid_df = pd.concat(dfs)


    print(fin_valid_df.shape)
    fin_valid_df.to_csv(config.LR_MODEL_PRED,index=False)