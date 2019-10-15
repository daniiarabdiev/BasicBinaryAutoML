import pandas as pd
from .utils import *
import os


def pipe_train(df, ID_COLUMN, LABEL_COLUMN, CATEGORICAL_COLUMNS, NUMERICAL_MEDIAN, key):
    # TRAIN
    df.set_index(ID_COLUMN, inplace=True)

    df_features = df.drop(LABEL_COLUMN, axis=1)
    target = df[[LABEL_COLUMN]].copy()
    print(target)
    df_features = impute_categorical(df_features, CATEGORICAL_COLUMNS)
    df_features = impute_numerical_median(df_features, NUMERICAL_MEDIAN)

    oof_pred_df = big_catboost(df_features, target, LABEL_COLUMN, CATEGORICAL_COLUMNS, key, False)
    if not os.path.exists('files/{0}/oof_preds'.format(key)):
        os.makedirs('files/{0}/oof_preds'.format(key))
    oof_pred_df.to_csv('files/{0}/oof_preds/oof_preds.csv'.format(key), index=False)


def pipe_predict(df_features_test, ID_COLUMN, CATEGORICAL_COLUMNS, NUMERICAL_MEDIAN, WEIGHTS, key):
    # PREDICTION
    df_features_test.set_index(ID_COLUMN, inplace=True)

    df_features_test = impute_categorical(df_features_test, CATEGORICAL_COLUMNS)
    df_features_test = impute_numerical_median(df_features_test, NUMERICAL_MEDIAN)

    pred_df = big_catboost_predict(df_features_test, WEIGHTS)

    if not os.path.exists('files/{0}/results'.format(key)):
        os.makedirs('files/{0}/results'.format(key))
    pred_df.to_csv('files/{0}/results/results.csv'.format(key), index=False)
    return pred_df
