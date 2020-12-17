from risk import *

"""
On fait l'hypothèse que: Les rendements plus élevés au cours des 12 derniers mois (252 jours)
sont proportionnels au rendement futur

"""
from zipline.pipeline.factors import Returns

def momentum_1yr(window_length, universe, sector):
    return Returns(window_length=window_length, mask=universe) \
        .demean(groupby=sector) \
        .rank() \
        .zscore()

"""
Implémentation de la fonction mean_reversion_5day_sector_neutral en supposant que
les surperformateurs à court terme (sous-performeurs) par rapport à leur secteur vont faire des retours.
Nous utilisons les données de retour de "universe" et rabaissons les données du secteur
pour partitionner, classer, puis nous les convertissons en zscore.

"""
def mean_reversion_5day_sector_neutral(window_length, universe, sector):
    """
    Génération d'un facteur neutre de secteur de retour à la moyenne sur 5 jours

    Paramètres
    ----------
    window_length : int
        Retourne les longueurs des fenêtres
    universe : Zipline Filter
        Univers de filtre des stocks
    sector : Zipline Classifier
        Classifieur de secteur

    Returns
    -------
    factor : Zipline Factor
        Facteur neutre de secteur de réversion moyenne sur 5 jours
    """

    factor = -(Returns(window_length=window_length, mask=universe) \
        .demean(groupby=sector) \
        .rank() \
        .zscore())

    return factor


project_tests.test_mean_reversion_5day_sector_neutral(mean_reversion_5day_sector_neutral)

#Visualisation des données de facteurs
factor_start_date = universe_end_date - pd.DateOffset(years=2, days=2)
sector = project_helper.Sector()
window_length = 5

pipeline = Pipeline(screen=universe)
pipeline.add(
    mean_reversion_5day_sector_neutral(window_length, universe, sector),
    'Mean_Reversion_5Day_Sector_Neutral')
engine.run_pipeline(pipeline, factor_start_date, universe_end_date)

from zipline.pipeline.factors import SimpleMovingAverage

"""
Facteur de réversion moyenne sur 5 jours de secteur neutre(lissé)
"""
def mean_reversion_5day_sector_neutral_smoothed(window_length, universe, sector):
    """
    Generate the mean reversion 5 day sector neutral smoothed factor

    Paramètres
    ----------
    window_length : int
        Retourne la longueur de la fenêtre
    universe : Zipline Filter
        Univers des filtres des stocks
    sector : Zipline Classifier
        Classifieur de secteurs

    Returns
    -------
    factor : Zipline Factor
        Facteur de réversion moyenne sur 5 jours de secteur neutre(lissé)
    """

    unsmoothed_factor = mean_reversion_5day_sector_neutral(window_length, universe, sector)
    factor = SimpleMovingAverage(inputs=[unsmoothed_factor],
                                 window_length=window_length)\
        .rank()\
        .zscore()
    return factor


project_tests.test_mean_reversion_5day_sector_neutral_smoothed(mean_reversion_5day_sector_neutral_smoothed)

#Visualition des données lissées
pipeline = Pipeline(screen=universe)
pipeline.add(
    mean_reversion_5day_sector_neutral_smoothed(5, universe, sector),
    'Mean_Reversion_5Day_Sector_Neutral_Smoothed')
engine.run_pipeline(pipeline, factor_start_date, universe_end_date)


"""
Facteur de sentiment Overnight
"""

from zipline.pipeline.data import USEquityPricing


class CTO(Returns):
    """
    Calcule le rendement Overnight, par hypothèse
    Code tiré du papier: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554010
    """
    inputs = [USEquityPricing.open, USEquityPricing.close]

    def compute(self, today, assets, out, opens, closes):
        """
        La matrice d'ouverture et de fermeture est de 2 lignes x N actifs avec le plus récent en bas.
        En tant que tel, opens[-1] est l'ouverture la plus récente et closes[0] est la fermeture la plus matinale
        """
        out[:] = (opens[-1] - closes[0]) / closes[0]


class TrailingOvernightReturns(Returns):
    """
    Somme des trailing 1m O/N returns
    """
    window_safe = True

    def compute(self, today, asset_ids, out, cto):
        out[:] = np.nansum(cto, axis=0)


def overnight_sentiment(cto_window_length, trail_overnight_returns_window_length, universe):
    cto_out = CTO(mask=universe, window_length=cto_window_length)
    return TrailingOvernightReturns(inputs=[cto_out], window_length=trail_overnight_returns_window_length) \
        .rank() \
        .zscore()

#On lisse le facteur précédemment claculé
def overnight_sentiment_smoothed(cto_window_length, trail_overnight_returns_window_length, universe):
    unsmoothed_factor = overnight_sentiment(cto_window_length, trail_overnight_returns_window_length, universe)
    return SimpleMovingAverage(inputs=[unsmoothed_factor], window_length=trail_overnight_returns_window_length) \
        .rank() \
        .zscore()

"""
Combinons les facteurs dans une seule pipeline

"""

universe = AverageDollarVolume(window_length=120).top(500)
sector = project_helper.Sector()

pipeline = Pipeline(screen=universe)
pipeline.add(
    momentum_1yr(252, universe, sector),
    'Momentum_1YR')
pipeline.add(
    mean_reversion_5day_sector_neutral(5, universe, sector),
    'Mean_Reversion_5Day_Sector_Neutral')
pipeline.add(
    mean_reversion_5day_sector_neutral_smoothed(5, universe, sector),
    'Mean_Reversion_5Day_Sector_Neutral_Smoothed')
pipeline.add(
    overnight_sentiment(2, 5, universe),
    'Overnight_Sentiment')
pipeline.add(
    overnight_sentiment_smoothed(2, 5, universe),
    'Overnight_Sentiment_Smoothed')
all_factors = engine.run_pipeline(pipeline, factor_start_date, universe_end_date)

all_factors.head()
