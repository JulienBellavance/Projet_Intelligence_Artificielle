"""
Module dédié au téléchargement et à l'entrainement des modèles
"""
import data_process
import numpy as nd
import pickle
from math import floor
import pandas as pd
from matplotlib import pyplot as pp
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

train_sets = data_process.noms_dataset
RAW_DATA_PATH = data_process.RAW_DATA_PATH
PROCESSED_DATA_PATH = data_process.PROCESSED_DATA_PATH

def optimize_lin(X, y):
    fit_intercept = [True, False]
    best_clf = LinearRegression().fit(X, y)
    best_score = best_clf.score(X,y)
    best_params = {'fit_intercept': True}

    for val in fit_intercept:
        clf = LinearRegression(fit_intercept=val)
        clf.fit(X,y)
        score = clf.score(X,y)

        if score > best_score:
            best_clf = clf
            best_params = {'fit_intercept': val}
            best_score = score

    return best_clf, best_params

def optimize_SVR(X, y):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    C = [0.4, 0.8, 1.0, 1.2, 1.6, 2.0]
    epsilon = [0.0001, 0.001, 0.01, 0.1, 1]
    best_clf = SVR()
    best_clf.fit(X, y)
    best_score = best_clf.score(X, y)
    best_params = {'kernel': 'rbf', 'C':0, 'epsilon':0.1}

    for k in kernels:
        for e in epsilon:
            for c in C:
                clf = SVR(kernel=k, C=c, epsilon=e)
                clf.fit(X,y)
                score = clf.score(X,y)

                if score > best_score:
                    best_clf = clf
                    best_params = {'kernel': k, 'C':c, 'epsilon':e}
                    best_score = score
    return best_clf, best_params

def optimize_kn(X, y):
    nNeighbours = [3, 4, 5, 6, 8, 10]
    algo = ['ball_tree', 'kd_tree']
    leaf_size = [10, 20, 25, 30, 35, 40, 50]
    best_clf = SVR()
    best_clf.fit(X, y)
    best_score = best_clf.score(X, y)
    best_params = {'n_neighbors': 5, 'algorithm': 'auto', 'leaf_size': 30}

    for a in algo:
        for n in nNeighbours:
            for l in leaf_size:
                clf = KNeighborsRegressor(n_neighbors=n, algorithm=a, leaf_size=l)
                clf.fit(X, y)
                score = clf.score(X, y)

                if score > best_score:
                    best_clf = clf
                    best_params = {'n_neighbors': n, 'algorithm': a, 'leaf_size': l}
                    best_score = score
    return best_clf, best_params

def optimize_rf(X, y):
    n_estimators = [50, 100, 150, 200]
    max_features = range(1, X.shape[1])
    best_clf = RandomForestRegressor()
    best_clf.fit(X, y)
    best_score = best_clf.score(X, y)
    best_params = {'n_estimators': 100, 'max_features': 'auto'}

    for n in n_estimators:
        for m in max_features:
            clf = RandomForestRegressor(n_estimators=n, max_features=m)
            clf.fit(X, y)
            score = clf.score(X, y)

            if score > best_score:
                best_clf = clf
                best_params = {'n_estimators': n, 'max_features': m}
                best_score = score
    return best_clf, best_params

def optimisation_parametres():
    if not Path("models").exists():
        Path.mkdir(Path("models"))
    linDone = False
    SVRDone = False
    knDone = False
    rdFDone = False

    data = data_process.genere_tab_commun()
    targets = data_process.classement_initial()
    scaled_data = minmax_scale(data, (-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, targets)

    if not linDone:
        lin, linParams = optimize_lin(X_train, y_train)
        pickle.dump(lin, open(Path("models/lin.sav"), 'wb'))
        pd.DataFrame(linParams, index=linParams.keys()).to_csv("models/linparams.csv")
        lin_params = nd.array(linParams)
    else:
        lin = pickle.load(open(Path("models/lin.sav"), 'rb'))
        lin_params = nd.genfromtxt('models/linparams.csv', delimiter=',')

    if not SVRDone:
        svr, svr_params = optimize_SVR(X_train, y_train)
        pickle.dump(svr, open(Path("models/svr.sav"), 'wb'))
        pd.DataFrame(svr_params, index=svr_params.keys()).to_csv("models/svrparams.csv")
        svr_params = nd.array(svr_params)
    else:
        svr = pickle.load(open(Path("models/svr.sav"), 'rb'))
        svr_params = nd.genfromtxt('models/svrparams.csv', delimiter=',')

    if not knDone:
        kn, kn_params = optimize_kn(X_train, y_train)
        pickle.dump(kn, open(Path("models/kn.sav"), 'wb'))
        pd.DataFrame(kn_params, index=kn_params.keys()).to_csv("models/knparams.csv")
        kn_params = nd.array(kn_params)
    else:
        kn = pickle.load(open(Path("models/kn.sav"), 'rb'))
        kn_params = nd.genfromtxt('models/knparams.csv', delimiter=',')

    if not rdFDone:
        rf, rf_params = optimize_rf(X_train, y_train)
        pickle.dump(rf, open(Path("models/rf.sav"), 'wb'))
        pd.DataFrame(rf_params, index=rf_params.keys()).to_csv("models/rfparams.csv")
        rf_params = nd.array(rf_params)
    else:
        rf = pickle.load(open(Path("models/rf.sav"), 'rb'))
        rf_params = nd.genfromtxt('models/rfparams.csv', delimiter=',')


    lin_score = lin.score(X_test, y_test)
    svr_score = svr.score(X_test, y_test)
    kn_score = kn.score(X_test, y_test)
    rf_score = rf.score(X_test, y_test)

    print(lin_score)
    print(svr_score)
    print(kn_score)
    print(rf_score)

if __name__ == "__main__":
    optimisation_parametres()