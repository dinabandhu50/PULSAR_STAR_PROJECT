import os 

# data path management
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DATA_PATH = PROJECT_ROOT_DIR + "/input/pulsar_data_train.csv"
TEST_DATA_PATH = PROJECT_ROOT_DIR + "/input/pulsar_data_test.csv"
TRAIN_FOLDS = PROJECT_ROOT_DIR + "/input/train_folds.csv" 

# model preds 
LR_MODEL_PRED = PROJECT_ROOT_DIR + "/model_preds/lr_preds.csv"
RF_MODEL_PRED = PROJECT_ROOT_DIR + "/model_preds/rf_preds.csv"

