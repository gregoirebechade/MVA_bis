import numpy as np
import sklearn as skl
import os 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from scipy.signal import welch
import scipy
from sklearn.ensemble import AdaBoostClassifier
from scipy.stats import kurtosis
from scipy.stats import skew
import xgboost as xgb

import json


print('début', flush=True)

with open('./radars/train_labels.json') as f: 
    dict_labels = json.load(f)

with open('./radars/test_labels.json') as f: 
    dict_labels_test = json.load(f)



np.random.seed(0)
train_set=[]
validation_set=[]
for i in range(2000): 
    uniform = np.random.uniform()
    if uniform < 0.8:
        train_set.append(i)
    else:
        validation_set.append(i)



def load_npz(file_path): 
    pdws = np.load(file_path)
    dates = pdws['date']
    largeurs = pdws['largeur']
    frequences=pdws['frequence']
    puissances = pdws['puissance']
    theta = pdws['theta']
    phi = pdws['phi']
    df=pd.DataFrame({'date':dates,'largeur':largeurs,'frequence':frequences,'puissance':puissances,'theta':theta,'phi':phi})
    return df

def get_means(df): 
    return pd.Series({'mean_date':np.mean(df['date']),'largeur': np.median(df['largeur']), 'mean_frequence':np.mean(df['frequence']),'mean_puissance':np.mean(df['puissance']),'mean_theta':np.mean(df['theta']),'mean_phi':np.mean(df['phi'])})

def get_quantiles (df):
    q1_date, q2_date = scipy.stats.mstats.mquantiles(df['date'], [0.25, 0.75])
    q1_frequence, q2_frequence = scipy.stats.mstats.mquantiles(df['frequence'], [0.25, 0.75])
    q1_theta, q2_theta = scipy.stats.mstats.mquantiles(df['theta'], [0.25, 0.75])
    q1_phi, q2_phi = scipy.stats.mstats.mquantiles(df['phi'], [0.25, 0.75])
    return pd.Series({'quantile_025_date': q1_date, 'quantile_075_date': q2_date, 'quantile_025_frequence': q1_frequence, 'quantile_075_frequence': q2_frequence, 'quantile_025_theta': q1_theta, 'quantile_075_theta': q2_theta, 'quantile_025_phi': q1_phi, 'quantile_075_phi': q2_phi})

def get_std(df): 
    return pd.Series({'std_date':np.std(df['date']),'std_frequence':np.std(df['frequence']),'std_puissance':np.std(df['puissance']),'std_theta':np.std(df['theta']),'std_phi':np.std(df['phi'])})

def get_kurtosis(df): 
    return pd.Series({'kurtosis_date':kurtosis(df['date']),'kurtosis_frequence':kurtosis(df['frequence']),'kurtosis_puissance':kurtosis(df['puissance']),'kurtosis_theta':kurtosis(df['theta']),'kurtosis_phi':kurtosis(df['phi'])})


def get_skew(df):
    return pd.Series({'skew_date':skew(df['date']),'skew_frequence':skew(df['frequence']),'skew_puissance':skew(df['puissance']),'skew_theta':skew(df['theta']),'skew_phi':skew(df['phi'])})


def reg_lin_theta(df):
    theta_permanent=df['theta'][int(0.2*len(df['theta'])):]
    X=np.arange(len(theta_permanent)).reshape(-1,1)
    y=theta_permanent
    reg=np.polyfit(X.flatten(), y, 1)
    a, b = reg
    return pd.Series({'a_reg_lin_theta':a, 'b_reg_lin_theta':b})


def Pxx_0_30(df): 
    dates=np.array(df['date'])
    puissances=np.array(df['puissance'])
    temps=np.arange(dates[0], dates[-1], 0.01)
    puiss_interp=np.interp(temps, dates, puissances)
    _, Pxx = welch(puiss_interp, fs=50, nperseg=128)
    return pd.Series({f'Pxx_{i}':Pxx[i] for i in range(len(Pxx[:30]))})


def maxs_pxx_0_30(df): 
    dates=np.array(df['date'])
    puissances=np.array(df['puissance'])
    temps=np.arange(dates[0], dates[-1], 0.01)
    puiss_interp=np.interp(temps, dates, puissances)
    _, Pxx = welch(puiss_interp, fs=50, nperseg=128)
    Pxx=Pxx/abs(Pxx)
    return  pd.Series({'max_pxx' : np.max(Pxx)})

def nombre_de_points(df): 
    return pd.Series({'nombre de points' : len(np.array(df['date']))})


def build_train_set(list_of_feature_extraction_functions): 
    X_train = []
    y_train = []
    for i in train_set: 
        file_path = f'./radars/train/pdw-{i}.npz'
        df = load_npz(file_path)
        features = pd.Series({'test':7})
        for function in list_of_feature_extraction_functions: 
            
            serie=function(df)
            features = pd.concat([features, serie])
        X_train.append(features)
        if dict_labels[f'pdw-{i}'] == 'menace': 
            lab=1
        else : 
            lab=0
        y_train.append(lab)
    X_train = pd.DataFrame(X_train)
    X_train.drop(columns=['test'],inplace=True)
    y_train = np.array(y_train)

    return X_train, y_train

def build_validation_set(list_of_feature_extraction_functions): 
    X_val = []
    y_val = []
    for i in validation_set: 
        file_path = f'./radars/train/pdw-{i}.npz'
        df = load_npz(file_path)
        features = pd.Series({'test':7})
        for function in list_of_feature_extraction_functions: 
            
            serie=function(df)
            features = pd.concat([features, serie])
        X_val.append(features)
        if dict_labels[f'pdw-{i}'] == 'menace': 
            lab=1
        else : 
            lab=0
        y_val.append(lab)
    X_val = pd.DataFrame(X_val)
    X_val.drop(columns=['test'],inplace=True)
    y_val = np.array(y_val)

    return X_val, y_val

def build_test_set(list_of_feature_extraction_functions):
    X_test = []
    y_test = []
    for i in range(100): 
        file_path = f'./radars/test/pdw-{i}.npz'
        df = load_npz(file_path)
        features = pd.Series({'test':7})
        for function in list_of_feature_extraction_functions: 
            serie=function(df)
            features = pd.concat([features, serie])
        X_test.append(features)
        if dict_labels_test[f'pdw-{i}'] == 'menace': 
            lab=1
        else : 
            lab=0
        y_test.append(lab)
    X_test = pd.DataFrame(X_test)
    X_test.drop(columns=['test'],inplace=True)
    y_test = np.array(y_test)

    return X_test, y_test




def train_model(model,  X_train, y_train): # entrainement de notre modèle sur un dataset fait de certaines features
    model.fit(X_train, y_train)
    return model

def predictions(model_trained, list_of_feature_extraction_functions, X_val):  # on prédit les labels de notre dataset de validation
    X_val, _ = build_validation_set(list_of_feature_extraction_functions)
    y_pred = model_trained.predict(X_val)
    return y_pred

def compute_confusion_matrix(y_val, y_pred): 
    return confusion_matrix(y_val, y_pred)
def compute_precision_recall(y_pred, y_val): # takes 2 arrays as an input and returns precision and recall
    precision = precision_score(y_val, y_pred, average='binary')
    recall = recall_score(y_val, y_pred, average='binary')
    return precision, recall


def evaluate_perf(list_of_feature_extraction_functions, model, X_train, y_train, X_val, y_val):
    model_trained = train_model(model,  X_train, y_train)
    y_pred = predictions(model_trained, list_of_feature_extraction_functions, X_val)
    precision, recall = compute_precision_recall(y_pred, y_val)
    acc = 0 
    y_val=np.array(y_val).flatten()
    y_pred=np.array(y_pred).flatten()
    for i in range(len(y_val)): 
        if y_val[i] == y_pred[i]:
            acc+=1
    acc = acc/len(y_val)
    return {'confusion matrix' : confusion_matrix(y_val, y_pred), 'precision': precision, 'recall':  recall, 'accuracy': acc}





features_list=[get_means, get_std, get_kurtosis, get_skew, reg_lin_theta, Pxx_0_30, maxs_pxx_0_30, nombre_de_points, get_quantiles]
X_val, y_val = build_validation_set(features_list)
X_train, y_train = build_train_set(features_list)
X_train_scaled = StandardScaler().fit_transform(X_train)
X_val_scaled = StandardScaler().fit_transform(X_val)


results_file='results_bis.txt'



for n_estimator in [10, 50, 100, 200, 500]: 
    ada=AdaBoostClassifier(n_estimators=n_estimator, random_state=42)
    res_ada=evaluate_perf(features_list, ada, X_train, y_train, X_val, y_val)
    res_ada_scaled=evaluate_perf(features_list, ada, X_train_scaled, y_train, X_val_scaled, y_val)
    with open(results_file, 'a') as f:
        f.write(f"n_estimators: {n_estimator}, res_ada: {res_ada}, res_ada_scaled: {res_ada_scaled}\n")
    for max_depth in [2,  4, 6, 8, 10, 15, 20]:
        xgboost=xgb.XGBClassifier( max_depth=max_depth)
        res_xgboost=evaluate_perf(features_list, xgboost, X_train, y_train, X_val, y_val)  
        res_xgb_scaled=evaluate_perf(features_list, xgboost, X_train_scaled, y_train, X_val_scaled, y_val) 
        with open(results_file, 'a') as f:
            f.write( f"max_depth: {max_depth}, n_estimator: {n_estimator},  res_xgboost: {res_xgboost}, res_xgb_scaled: {res_xgb_scaled}\n")
        print(f"n_estimators: {n_estimator}, max_depth: {max_depth}")
        rf=RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth, random_state=42)
        gb=GradientBoostingClassifier(n_estimators=n_estimator, max_depth=max_depth, random_state=42)
        res_gb=evaluate_perf(features_list, gb, X_train, y_train, X_val, y_val)
        res_rf=evaluate_perf(features_list, rf, X_train, y_train, X_val, y_val)
        res_gb_scaled=evaluate_perf(features_list, gb, X_train_scaled, y_train, X_val_scaled, y_val)
        res_rf_scaled=evaluate_perf(features_list, rf, X_train_scaled, y_train, X_val_scaled, y_val)
        with open(results_file, 'a') as f: 
            f.write(f"n_estimators: {n_estimator}, max_depth: {max_depth}, res_gb: {res_gb}, res_rf: {res_rf}, res_gb_scaled: {res_gb_scaled}, res_rf_scaled: {res_rf_scaled}\n")



print('fini !')