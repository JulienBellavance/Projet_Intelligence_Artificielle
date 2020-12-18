"""
Module dédié à la préparation des données
"""
import numpy as nd
import pandas as pd
import quandl
from pathlib import Path
import fnmatch
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


DATE_DEBUT = "2017-12-31"
DATE_FIN = "2019-12-31"
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
SPLIT_MULTIPLIER = 0.5

###########################################################################
"""
Configuration de l'api Sharadar. Ne pas changer la clé.
Pour utiliser l'api: 
    -Pour obtenir la valeur d'une action dans le temps: 
    
        quandl.get("CODE_DE_L'ACTION", [start_date="date_de_debut", end_date="date_de_fin", collapse="frequence", return="numpy"])
        
            "CODE_DE_L'ACTION": Le code représentant l'action dans la bases de données
            start_date, end_date: optionnels, filtre la base de données pour ne retrouner que les valeurs entre ces deux dates
            collapse: optionnel, détermine l'intervalle entre les valeurs retournées (défaut = daily)
            return="numpy": optionnel, retourne les valeurs demandées sous la forme d'un array Numpy
    
       
    
Les Codes des différentes actions sont disponibles sur le site de Quandl
"""
CLE_QUANDL = "r6aerBdvstyRvzzhYapz"
quandl.ApiConfig.api_key = CLE_QUANDL
###########################################################################

"""
Sélection des données à récupérer
"""
#Datasets d'entrainement
noms_dataset = ["Apple QuoteMedia End of Day US Prices",
                "Microsoft QuoteMedia End of Day US Prices",
                "Intel QuoteMedia End of Day US Prices",
                "Verizon QuoteMedia End of Day US Prices",
                "United Technologies QuoteMedia End of Day US Prices",
                "Exxon Mobile QuoteMedia End of Day US Prices",
                "Visa QuoteMedia End of Day US Prices",
                "United Health Group QuoteMedia End of Day US Prices",
                "General Electric QuoteMedia End of Day US Prices",
                "Walt Disney QuoteMedia End of Day US Prices"
                ]
codes_dataset = ["EOD/AAPL",
                 "EOD/MSFT",
                 "EOD/INTC",
                 "EOD/VZ",
                 "EOD/UTX",
                 "EOD/XOM",
                 "EOD/V",
                 "EOD/UNH",
                 "EOD/GE",
                 "EOD/DIS"
                 ]

#validation des dossiers de données
def valider_paths():
    if not Path("data").exists():
        Path.mkdir(Path("data"))
    if not Path(RAW_DATA_PATH).exists():
        Path.mkdir(Path(RAW_DATA_PATH))
    if not Path(PROCESSED_DATA_PATH).exists():
        Path.mkdir(Path(PROCESSED_DATA_PATH))
    return

#téléchargement des données brutes
def telecharger_donnees(nom, code, training=False):
    if training:
        for i in range(len(codes_dataset)):
            if not Path(RAW_DATA_PATH + noms_dataset[i] + ".csv").exists():
                print("Téléchargement de " + RAW_DATA_PATH + noms_dataset[i])
                data = quandl.get(codes_dataset[i])
                data.to_csv(Path(RAW_DATA_PATH + noms_dataset[i] + ".csv"))
        return

    if not Path(RAW_DATA_PATH + nom + ".csv").exists():
        print("Téléchargement de " + RAW_DATA_PATH + nom)
        data = quandl.get(code)
        data.to_csv(Path(RAW_DATA_PATH + nom + ".csv"))


#Pre-processing
def data_processing(_datasets, training=False):
    if training: datasets = noms_dataset
    else: datasets = _datasets

    for dataset in datasets:
        if not Path(PROCESSED_DATA_PATH + dataset + ".csv").exists():
            data = pd.read_csv(Path(RAW_DATA_PATH + dataset + ".csv"), index_col=0)

            #Calcul du rendement
            valeur_close = []
            gain = []
            rendement = []
            rentabilite = []

            for i in range(data["Close"].size - 1):
                valeur_close.append(data["Close"].values[i + 1])
                gain.append(data["Close"].values[i + 1] - data["Close"].values[i])
                gain_adj = data["Adj_Close"].values[i + 1] - data["Adj_Close"].values[i]
                facteur = data["Adj_Close"].values[i + 1] / (SPLIT_MULTIPLIER * data["Close"].values[i + 1])
                dividende = (1 - facteur) * gain[i]
                rendement.append((dividende/data["Close"].values[i + 1])/data["Close"].values[i])
                rentabilite.append(rendement[i] + (gain[i]/data["Close"].values[i]))

            index = data.index[1:]
            data = {'Valeurs': valeur_close, 'Gain': gain, 'Rendement': rendement, 'Rentabilite': rentabilite}

            dtframe = pd.DataFrame(data=data, index=index)
            dtframe.to_csv(Path(PROCESSED_DATA_PATH + dataset + "_processed.csv"))


# generation du tableau commun
def genere_tab_commun():
    #commun = pd.DataFrame(columns={"Valeurs", "Gain", "Rendement", "Rentabilite"})
    commun=[]
    for dataset in noms_dataset:
        donnees = pd.read_csv(Path(PROCESSED_DATA_PATH + dataset + "_processed.csv"), index_col=0)
        #print(donnees)
        #commundf = pd.read_table(donnees, delim_whitespace=True,
        #                         names={"Date", "Valeurs", "Gain", "Rendement", "Rentabilite"})
        #commun.append(donnees, ignore_index=False)
        commun.append(donnees)
    #print(commun[0]["Valeurs"][0])
    return commun

# generation du classement initial
def classement_initial():
    commun = genere_tab_commun()
    # classement initial
    liste_classe = nd.ndarray((10, 1089))
    for i in range(len(commun)):
        #print(len(commun[i]["Rentabilite"]))
        for j in range(len(commun[i]["Rentabilite"])):
            if commun[i]["Rentabilite"][j]>0:
                liste_classe[i][j] = 1 #bon investissement
            else:
                liste_classe[i][j] = 0 #mauvais investissement
        #print(len(classe) 
    #print(liste_classe)
    #print(len(liste_classe))
    #print(len(liste_classe[1]))
    return liste_classe


# prediction des classes
def prediction():
    # liste des classifieurs
    classifieurs = [LinearRegression,
                SVR,
                KNeighborsRegressor,
                RandomForestRegressor
        ]
    class_initial = classement_initial()
    commun = genere_tab_commun()
    rentabilite = nd.ndarray((10, 1089))
    for i in range(len(commun)):
        rentabilite[i] = commun[i]["Rentabilite"]
    #print(rentabilite)
    #predictions = nd.zeros((len(classifieurs)))
    predictions = nd.ndarray((10, 1089))
    classif = []
    for classifiers in classifieurs:
        for i in range(len(rentabilite)):
            clf = classifiers()
            clf.fit(rentabilite[i].reshape(-1,1), class_initial[i])
            #predictions[i] = clf.predict(rentabilite[i].reshape(-1,1))
            predictions[i] = clf.predict(class_initial[i].reshape(-1, 1))
        classif.append(predictions)
    print("classement initial: {}".format(class_initial))
    print("rentabilite : {}".format(rentabilite))
    print("prediction : {}".format(classif))
    return

if __name__ == "__main__":
    valider_paths()
    telecharger_donnees(None, None, training=True)
    data_processing(None, training=True)
    genere_tab_commun()
    classement_initial()
    prediction()

