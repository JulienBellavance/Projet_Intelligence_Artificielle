"""
Module dédié à la préparation des données
"""
import numpy as nd
import pandas as pd
import quandl
import os
from pathlib import Path
import fnmatch


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

if not Path("data").exists():
    Path.mkdir(Path("data"))
if not Path(RAW_DATA_PATH).exists():
    Path.mkdir(Path(RAW_DATA_PATH))
if not Path(PROCESSED_DATA_PATH).exists():
    Path.mkdir(Path(PROCESSED_DATA_PATH))

#téléchargement des données brutes
for i in range(len(codes_dataset)):
    if not Path(RAW_DATA_PATH + noms_dataset[i] + ".csv").exists():
        print("Téléchargement de " + RAW_DATA_PATH + noms_dataset[i])
        data = quandl.get(codes_dataset[i])
        chemin = data.to_csv(Path(RAW_DATA_PATH + noms_dataset[i] + ".csv"))


#Pre-processing
for dataset in noms_dataset:
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
commun = pd.DataFrame(columns={"Date", "Valeurs", "Gain", "Rendement", "Rentabilite"})
for dataset in  noms_dataset:
    donnees = open(Path(PROCESSED_DATA_PATH + dataset + "_processed.csv"), 'r')
    commundf = pd.read_table(donnees, delim_whitespace=True, names={"Date", "Valeurs", "Gain", "Rendement", "Rentabilite"})
    commun.append(commundf, ignore_index=False)
print(commun)