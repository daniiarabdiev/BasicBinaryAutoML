def preedit():
    import pandas as pd
    df = pd.read_csv('occidental_admission_data.csv')
    df['id'] = df.index
    df.drop('Extracurricular Interests', axis=1, inplace=True)
    train=df.sample(frac=0.8,random_state=200) #random state is a seed value
    test=df.drop(train.index)
    test[['Gross Commit Indicator']].to_csv('label_test.csv')
    test.drop('Gross Commit Indicator', axis=1, inplace=True)
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)
    print(train.shape)
    print(test.shape)


def impute_categorical(df, CATEGORICAL_COLUMNS):
    df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].astype(object)
    df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].fillna('-999')
    return df


def impute_numerical_median(df, NUMERICAL_MEDIAN):
    for col in NUMERICAL_MEDIAN:
        df[col] = df[col].fillna(df[col].median())
    return df


def classification_metrics(y_true, y_pred, y_pred_prob):
    '''
    The function that calculates different metrics related to classification like confusion metric, accuracy,
    AUC and Precision, Recall, F1 scores.
    :param y_true:
    :param y_pred:
    :param y_pred_prob:
    :return:
    '''
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

    # Accuracy
    accuracy = round(accuracy_score(y_true, y_pred), 3)
    print("Accuracy - test set: %.2f%%" % (accuracy * 100.0))

    # Classification report
    class_report = classification_report(y_true, y_pred)
    print(class_report)

    # Construct the Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # AUC
    auc = round(roc_auc_score(y_true, y_pred_prob), 3)
    print('AUC {0}'.format(auc))

    return auc, accuracy


def catboost_feature_importance(model, train_pool, dataframe_columns):
    import pandas as pd
    feature_importances = model.get_feature_importance(train_pool)
    imp_feat = []
    for score, name in sorted(zip(feature_importances, dataframe_columns), reverse=True):
        if score > 0:
            imp_feat.append([name, score])

    imp_feat_df = pd.DataFrame(imp_feat, columns=['feature', 'score'])
    return imp_feat_df


def big_catboost(df_tr, target, label_column, cat_columns, key, undersample=False):
    from catboost import CatBoostClassifier, Pool
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    from sklearn.metrics import roc_auc_score
    import pandas as pd

    y_train = target[label_column]

    # CROSS VALIDATION
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    # RUN
    c = 0
    oof_preds = np.zeros((len(df_tr), 2))

    auc_scores = []

    for train, valid in cv.split(df_tr, y_train):
        print("VAL %s" % c)
        X_train = df_tr.iloc[train]
        Y_train = y_train.iloc[train]
        X_valid = df_tr.iloc[valid]
        Y_valid = y_train.iloc[valid]

        model = CatBoostClassifier(loss_function='Logloss',
                                   eval_metric='AUC',
                                   logging_level='Verbose',
                                   random_seed=42,
                                   od_type='Iter',
                                   num_boost_round=50000,
                                   od_wait=300)

        if undersample is True:
            X_train, Y_train = random_undersample(X_train, Y_train, label_column)

        cat_columns_indices = [X_train.columns.get_loc(c) for c in cat_columns if c in X_train]
        model.fit(X_train, Y_train, eval_set=(X_valid, Y_valid), use_best_model=True,
                  cat_features=cat_columns_indices)
        oof_preds[valid] = model.predict_proba(X_valid)
        print(key)
        import os
        if not os.path.exists('files/{0}/weights'.format(key)):
            os.makedirs('files/{0}/weights'.format(key))
        if not os.path.exists('files/{0}/important_features'.format(key)):
            os.makedirs('files/{0}/important_features'.format(key))
        model.save_model('files/{0}/weights/catboost_model_{1}.dump'.format(key, c))

        # FEATURE IMPORTANCE
        train_pool = Pool(X_train, Y_train, cat_features=cat_columns_indices)
        imp_feats = catboost_feature_importance(model, train_pool, X_train.columns)
        imp_feats.to_csv('files/{0}/important_features/important_features_{1}.csv'.format(key, c), index=False)
        # EVALUATION PHASE
        auc, accuracy = classification_metrics(Y_valid, model.predict(X_valid), [r[1] for r in oof_preds[valid]])
        c += 1
        auc_scores.append(auc)

    oof_preds = [row[1] for row in oof_preds]
    auc = roc_auc_score(y_train, oof_preds)
    print("CV_AUC: {}".format(auc))

    print('Best AUC: ', max(auc_scores))

    # SAVE OOF PREDS
    oof_pred_df = pd.DataFrame(columns=['ID_code', 'target'])
    oof_pred_df['ID_code'] = pd.Series(df_tr.index.tolist())
    oof_pred_df['target'] = pd.Series(oof_preds)
    return oof_pred_df


def big_catboost_predict(df_features, weights_folder):
    import os
    import pandas as pd
    from catboost import CatBoostClassifier
    import numpy as np

    preds = []
    for file in os.listdir(weights_folder):
        model = CatBoostClassifier(loss_function='Logloss',
                                   eval_metric='AUC',
                                   logging_level='Verbose',
                                   random_seed=42,
                                   od_type='Iter',
                                   num_boost_round=50000,
                                   od_wait=300)
        model.load_model(os.path.join(weights_folder, file))

        curr_preds = model.predict_proba(df_features)
        curr_preds = [row[1] for row in curr_preds]
        preds.append(curr_preds)

    preds = np.asarray(preds)
    preds = preds.reshape((5, df_features.shape[0]))
    preds_final = np.mean(preds.T, axis=1)
    pred_df = pd.DataFrame(columns=['id', 'predictions', 'probability'])
    pred_df['id'] = df_features.index
    pred_df['probability'] = preds_final
    pred_df['predictions'] = [1 if i > 0.5 else 0 for i in preds_final]

    return pred_df
