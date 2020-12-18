"""
Module dédié au téléchargement et à l'entrainement des modèles
"""
import data_process
import numpy as nd
import pickle
from math import floor
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

train_sets = data_process.noms_dataset
RAW_DATA_PATH = data_process.RAW_DATA_PATH
PROCESSED_DATA_PATH = data_process.PROCESSED_DATA_PATH
linDone = False
SVRDone = False
knDone = False
rdFDone = False

def optimize_lin(X_train, y_train):
    pass
def optimize_SVR(X_train, y_train):
    pass
def optimize_kn(X_train, y_train):
    pass
def optimize_rf(X_train, y_train):
    pass

def optimisation_parametres():
    data = pd.read_csv(Path(RAW_DATA_PATH + train_sets[0] + ".csv"), index_col=0).to_numpy()
    targets = data_process.classement_initial()[0][:1090]
    scaled_data = scale(data)

    tt = floor(2*len(targets)/3)

    X_train = data[:tt]
    X_test = data[tt:]
    y_train = targets[:tt]
    y_test = targets[tt:]

    if not linDone:
        optimize_lin(X_train, y_train)
    if not SVRDone:
        optimize_SVR(X_train, y_train)
    if not knDone:
        optimize_kn(X_train, y_train)
    if not rdFDone:
        optimize_rf(X_train, y_train)

    lr = LinearRegression()
    svr = SVR()
    kn = KNeighborsRegressor()
    rndForest = RandomForestRegressor()
    lr.fit(X_train, y_train)
    print(lr.score(X_test, y_test))

    svr.fit(X_train, y_train)
    print(svr.score(X_test, y_test))

    kn.fit(X_train, y_train)
    print(kn.score(X_test, y_test))

    rndForest.fit(X_train, y_train)
    print(rndForest.score(X_test, y_test))

if __name__ == "__main__":
    optimisation_parametres()