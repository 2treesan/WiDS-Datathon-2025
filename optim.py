import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import f1_score as f1_score_calc
from lightgbm import LGBMRegressor, LGBMClassifier
import argparse
import time
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import random
import multiprocessing
import os
SEED = 42
random.seed(42)
np.random.seed(SEED)

def F1(y_true, y_pred, threshold=0.5, weight=None):
    x = f1_score_calc(y_true, (y_pred > threshold).astype(int), sample_weight=weight)
    # print(x)
    return x

def balanced_kfold_split(df_xx, df_x, n_splits=1, group_column='ADHD_Outcome', df_buff=None, seed=42):
    """
    Custom balanced k-fold split for a dataframe ensuring balanced distribution for each groups
    Args:
        df_xx: DataFrame to be split
        df_x: Origin DataFrame containing the target variable
        n_splits: Number of splits
        group_column: Column name to be used for grouping
        df_buff: Buffer DataFrame to be used in all folds
    """
    if n_splits==1:
        return [[df_xx.index, df_xx.index]]

    # Align group column from df_x to df
    df = df_xx.copy()
    df[group_column] = df_x[group_column].values  # Ensure correct order of target in df
    buffer_indices = None
    if df_buff is not None:
        buffer_indices = df_buff.index.tolist()

    # Initialize folds
    folds = [[] for _ in range(n_splits)]
    groups = df[group_column].unique()

    for group in groups:
        group_indices = df[df[group_column] == group].index.tolist()
        np.random.seed(seed)
        np.random.shuffle(group_indices)  # Shuffle indices for the group

        # Distribute the group indices across folds
        for i, idx in enumerate(group_indices):
            folds[i % n_splits].append(idx)

    # Create train and validation indices for each fold
    splits = []
    for i in range(n_splits):
        val_indices = np.array(folds[i])
        train_indices = np.array([idx for fold in folds if fold != folds[i] for idx in fold])
        if buffer_indices:
            buffer_indices_tmp = [idx for idx in buffer_indices if idx not in train_indices]
            print('buff', len(buffer_indices_tmp))
            train_indices = np.array(list(train_indices)+buffer_indices_tmp)
        splits.append((train_indices, val_indices))
    np.random.seed(42)
    return splits

def optim_th(final_pred, gt=None):
    # gt = {'ADHD_Outcome': [1,1,1,1,...],
    #      'Sex_F': [1,1,1,1,...]
    # }

    # Compute F1 scores and find the best threshold
    thresholds = np.linspace(0, 1, 100)
    best_theshold_dict = {}
    best_score_dict = {}

    for idx, c in enumerate(CONFIG.target_cols):
        adhd_scores = []
        splits = range(1)
        for t in thresholds:
            x = 0
            for i in range(len(splits)):
                weights = [1 for _ in gt[c]]
                if len(final_pred.shape)==1:
                    final_pred = final_pred[:,None]
                tmp_score = F1(gt[c], final_pred[:len(gt[c])][:, idx], t, weight=weights)
                x += tmp_score
            adhd_scores.append(x/len(splits))
        max_score = max(adhd_scores)
        # boolean mask of where scores == max
        # we find list all threshold given best score
        mask = np.array(adhd_scores) == max_score
        best_theshold_dict[c] = list(thresholds[mask])
        # print('best_adhd_threshold', best_adhd_threshold)
        best_score_dict[c] = max_score
    return best_theshold_dict, best_score_dict

def train_fold(fold, n_jobs, train_index, val_index,
               feature_train, target_train, feature_test, weights_train, best_hypr, model_type='lgb',only_run_fold=None, global_parallel=8):

    if only_run_fold is not None and only_run_fold!=fold:
        print('SKIP FOLD')
        return None

    if 'Sex_F' not in CONFIG.target_cols:
        feature_train = feature_train.drop([col for col in feature_test.columns if 'throw' in col], axis=1)
        feature_test = feature_test.drop([col for col in feature_test.columns if 'throw' in col], axis=1)
        
    # Create train and validation sets
    X_train, X_val = feature_train.iloc[train_index], feature_train.iloc[val_index]
    y_train, y_val = target_train.iloc[train_index], target_train.iloc[val_index]
    # y_train, y_val = y_train, y_val
    weight_train, weight_val = weights_train.iloc[train_index], weights_train.iloc[val_index]
    
    if 'lgb' in model_type:

        model = LGBMRegressor(**best_hypr,
        # model = LGBMClassifier(**best_hypr,
                              objective='binary',
                              n_jobs=(multiprocessing.cpu_count()//n_jobs)//global_parallel,
                              random_state=42,
                              extra_trees=True,
                              # weight_column='weight',
                             )

    elif 'xgb' in model_type:
        model = XGBRegressor(objective='binary:logistic', **best_hypr,
        # model = XGBClassifier(objective='binary:logistic', **best_hypr,
                          device='cpu',
                        random_state=42,
                        enable_categorical=True,
                          nthread=(multiprocessing.cpu_count()//n_jobs)//global_parallel,
                        )

    elif model_type=='cat':
        model = CatBoostClassifier(loss_function="CrossEntropy", **best_hypr, # MultiLogloss, MultiCrossEntropy
            verbose=False,
            random_seed=42,
            thread_count=(multiprocessing.cpu_count()//n_jobs)//global_parallel,
            grow_policy='Lossguide',
            task_type = 'CPU',
            cat_features=list(X_train.select_dtypes(include=['category']).columns),

        )
    model = MultiOutputRegressor(model)
    buf = 0
    # # Train the model
    model.fit(X_train.drop(columns=CONFIG.target_cols), y_train.values,
             # sample_weight=weight_train['weight'].values.ravel()
             )

    y_pred_val = model.predict(X_val.drop(columns=CONFIG.target_cols))
    y_pred_test = model.predict(feature_test)
    # if fold==0:
    #     print(y_pred_test)
    return fold, y_pred_val, y_val, y_pred_test

# Objective function for Optuna, runs optimization for each fold
def objective(trial, feature_train, target_train, weights_train, feature_test, pseudo_test_label, splits, model_type='lgb', only_run_fold=None, global_parallel=8):
    if model_type=='lgb':
        best_hypr = {
            "verbosity": -1,
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1, step=0.01),
            "n_estimators": trial.suggest_categorical("n_estimators", list(range(200, 501, 100))),
            'max_leaves': trial.suggest_int('max_leaves', 6, 130),
            'min_child_samples': trial.suggest_int('min_child_samples', 7, 50),
        }
    elif model_type=='xgb':
        best_hypr = {
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1, step=0.01),
            "n_estimators": trial.suggest_categorical("n_estimators", list(range(100, 501, 10))),
            'max_depth': trial.suggest_categorical("max_depth", list(range(5, 11, 1))),
            'max_leaves': trial.suggest_int('max_leaves', 6, 130),
        }
    elif model_type=='cat':
        best_hypr = {
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1, step=0.01),
            "n_estimators": trial.suggest_categorical("n_estimators", list(range(100, 501, 100))),
            'max_depth': trial.suggest_categorical("max_depth", list(range(5, 12, 1))),
            'max_leaves': trial.suggest_int('max_leaves', 6, 130),
        }


    n_jobs_x = min(8, multiprocessing.cpu_count()//4) # ensure we have at least 4 cores for each fold
    # Use joblib to parallelize the fold training
    results = Parallel(n_jobs=n_jobs_x, backend='threading')(
        delayed(train_fold)(fold, min(len(splits), n_jobs_x), train_index, val_index,
                            feature_train.copy(deep=False), target_train.copy(), feature_test.copy(), weights_train,
                            best_hypr, model_type, only_run_fold, global_parallel
                           )
        for fold, (train_index, val_index) in enumerate(splits)
    )
    results = sorted(results, key=lambda x: x[0])

    # Aggregate predictions and scores

    best_threshold_oof = {k: [] for k in CONFIG.target_cols}
    best_score_oof = {k: 0 for k in CONFIG.target_cols}
    for i, fold_result in enumerate(results):
        idx, y_pred_val, y_val, y_pred_test = fold_result
        for i,col in enumerate(CONFIG.target_cols):
            best_threshold, best_score = optim_th(y_pred_val, {k: y_val.values[:,i] for i, k in enumerate(CONFIG.target_cols)})
            for k, v in best_threshold.items():
                best_threshold_oof[k].extend(v)
                best_score_oof[k] += best_score[k]
        
    if len(splits) == 1:
        # threshold optimizer to 0.5
        err = 0
        for col in CONFIG.target_cols:
            err += np.abs(0.5-np.mean(best_threshold_oof[col]))

        test_f1 = []
        # print(pseudo_test_label.values.shape)
        for i,col in enumerate(CONFIG.target_cols):
            test_f1.append(F1(pseudo_test_label.values[:,i], y_pred_test[:,i]))
            
        return (np.mean(list(best_score_oof.values()))+np.mean(test_f1))/2 - err*1.2
    else:
        return np.mean(list(best_score_oof.values()))


# Main function to handle 8-fold cross-validation and Optuna optimization
def run_optimization(feature_train, df, target_train, weights_train, feature_test, model_type='lgb', only_run_fold=None, global_parallel=8):
    
    # Initialize KFold for 8 splits
    splits = balanced_kfold_split(feature_train, df, n_splits=8, group_column='ADHD_Outcome')
    
    # splits = [[feature_train.index, feature_train.index]] # Use for whole data hyperparamters optimizer
    pseudo_test_label = None
    if len(splits)==1:
        print('Use pseudo label for whole data optimizer')
        sub = pd.read_csv('/kaggle/working/submission.csv')
        pseudo_test_label = sub[CONFIG.target_cols]
        
    print('NUM SPLITS:', len(splits))
    best_hyperparameters = []

    # Optimize hyperparameters for this fold using Optuna
    study = optuna.create_study(direction="maximize")

    study.optimize(lambda trial: objective(trial, feature_train, target_train, weights_train, feature_test, pseudo_test_label,
                                           splits, model_type, only_run_fold, global_parallel),
                   n_trials=10 if 'Sex_F' not in CONFIG.target_cols else 200 # Sex_F optimization run slow, so reduce n_trials
                  )

    best_hyperparameters.append(study.best_trial.params)
    print(f"Best params: {study.best_trial.params}")

    return best_hyperparameters

class CONFIG:
    th = {"ADHD_Outcome": 0.7, "Sex_F": 0.15}
    target_cols = ['ADHD_Outcome']
CONFIG.target_cols = ['Sex_F']
CONFIG.target_cols = ['ADHD_Outcome']
# CONFIG.target_cols = ['ADHD_Outcome', 'Sex_F']

if not os.path.isfile('/kaggle/working/feature_train.parquet'):
    print('File not found, make sure you save_out in get_data function.')
    
feature_train = pd.read_parquet("feature_train.parquet", engine="fastparquet")
feature_test = pd.read_parquet("feature_test.parquet", engine="fastparquet")
weights_train = pd.read_parquet("weights_train.parquet", engine="fastparquet")
df = pd.read_parquet("df.parquet", engine="fastparquet")
target_train = feature_train[CONFIG.target_cols].copy()

categorical_columns=[
        "Basic_Demos_Study_Site",
        "Basic_Demos_Study_Site",
        "MRI_Track_Scan_Location",
        "Basic_Demos_Enroll_Year",
        "PreInt_Demos_Fam_Child_Ethnicity",
        "PreInt_Demos_Fam_Child_Race",
        'Barratt_Barratt_P1_Occ',
        'Barratt_Barratt_P2_Occ',
        'Barratt_Barratt_P1_Edu',
        'Barratt_Barratt_P2_Edu',
        ]
combined = pd.concat([feature_train,feature_test],axis=0,ignore_index=True)
for col in categorical_columns:
    if col in feature_test.columns.tolist():
        combined[col] = combined[col].astype('category')
    else:
        print('No', col)

# print(len(combined))

feature_test = combined.iloc[len(feature_train):].reset_index(drop=True).copy()
# print(len(feature_test))
feature_train = combined.iloc[:len(feature_train)].copy()
feature_train.drop([x for x in ["ADHD_Outcome", "Sex_F"] if x not in CONFIG.target_cols and x in list(feature_train.columns)], axis=1, inplace=True)
feature_test.drop([x for x in ["ADHD_Outcome", "Sex_F"] if x in list(feature_test.columns)], axis=1, inplace=True)


if __name__ == "__main__":
    print('START')
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_run_fold", type=int, required=False, default=None, help="Fold number to run (0-7)")
    parser.add_argument("--global_parallel", type=int, required=True, default=None, help="Max parallel")
    parser.add_argument("--model_type", type=str, required=True, default='lgb', help="Model type")
    args = parser.parse_args()
    print('RUN CONFIG:\n', args)
    # Replace with your actual data (feature_train, target_train, feature_test)
    best_hyperparameters = run_optimization(feature_train, df, target_train, weights_train, feature_test,
                                            model_type=args.model_type,
                                            only_run_fold=args.only_run_fold,
                                           global_parallel=args.global_parallel)

    if args.only_run_fold is None:
        # Output the best hyperparameters for all folds

        print("Best hyperparameters:")
        print(best_hyperparameters)
        for idx, params in enumerate(best_hyperparameters):
            print(f"Fold {idx + 1}: {params}")
        # Save the results to a file
        with open(f'/kaggle/working/best_hyperparameters.txt', 'w') as f:
            for idx, params in enumerate(best_hyperparameters):
                f.writelines([f"Fold {idx + 1}:\n"])
                for key, value in params.items():
                    f.writelines([f"{key}: {value}\n"])
    else:
        print("Best hyperparameters for fold:", args.only_run_fold)
        print(f"Fold: {best_hyperparameters[0]}")
        # Save the results to a file
        with open(f'/kaggle/working/best_hyperparameters_fold_{args.only_run_fold}.txt', 'w') as f:
            for idx, params in enumerate(best_hyperparameters):
                f.writelines([f"Fold {args.only_run_fold}:\n"])
                for key, value in params.items():
                    f.writelines([f"{key}: {value}\n"])
