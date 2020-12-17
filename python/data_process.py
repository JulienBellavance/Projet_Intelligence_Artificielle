"""
Module dédié à la préparation des données
"""
import numpy
import quandl
from pathlib import Path


DATE_DEBUT = "2018-12-31"
DATE_FIN = "2019-12-31"
RAW_DATA_PATH = "../data/raw/"
PROCESSED_DATA_PATH = "../data/processed/"

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
noms_dataset = ["WTI Crude Oil Price",
                "Microsoft QuoteMedia End of Day US Prices"
                ]
codes_dataset = ["EIA/PET_RWTC_D",
                 "EOD/MSFT"
                 ]

for i in range(len(codes_dataset)):
    if not Path(RAW_DATA_PATH + noms_dataset[i] + ".csv").exists():
        data = quandl.get(codes_dataset[i], returns="numpy")
        numpy.savetxt(Path(RAW_DATA_PATH + noms_dataset[i] + ".csv"), data, delimiter=',')

l