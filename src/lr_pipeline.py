import numpy as np
from sklearn import pipeline
from sklearn import preprocessing 
from sklearn import impute

lr_pipe = pipeline.Pipeline([
    ('imp',impute.SimpleImputer(missing_values=np.nan,strategy="median")),
    ('scaler',preprocessing.StandardScaler())
])