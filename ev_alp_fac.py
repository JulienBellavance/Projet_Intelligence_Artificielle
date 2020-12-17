from cre_alp_fa import *

"""
Nous évaluons les facteurs alpha en utilisant un délai de 1

"""

import alphalens as al

assets = all_factors.index.levels[1].values.tolist()
pricing = get_pricing(
    data_portal,
    trading_calendar,
    assets,
    factor_start_date,
    universe_end_date)

#Pour utiliser les fonctions alphalens, nous devons aligner
#les indices et convertir les temps en timestamp unix

clean_factor_data = {
    factor: al.utils.get_clean_factor_and_forward_returns(factor=factor_data, prices=pricing, periods=[1])
    for factor, factor_data in all_factors.iteritems()}

unixt_factor_data = {
    factor: factor_data.set_index(pd.MultiIndex.from_tuples(
        [(x.timestamp(), y) for x, y in factor_data.index.values],
        names=['date', 'asset']))
    for factor, factor_data in clean_factor_data.items()}


#Permet de voir facteurs de retour temporellement
ls_factor_returns = pd.DataFrame()

for factor, factor_data in clean_factor_data.items():
    ls_factor_returns[factor] = al.performance.factor_returns(factor_data).iloc[:, 0]

(1+ls_factor_returns).cumprod().plot()

#Points de base par jour et par quantile
qr_factor_returns = pd.DataFrame()

for factor, factor_data in unixt_factor_data.items():
    qr_factor_returns[factor] = al.performance.mean_return_by_quantile(factor_data)[0].iloc[:, 0]

(10000*qr_factor_returns).plot.bar(
    subplots=True,
    sharey=True,
    layout=(4,2),
    figsize=(14, 14),
    legend=False)

#Analyse de turnover
#Afin de connaitre la stabilité des alphas à travers le temps

ls_FRA = pd.DataFrame()

for factor, factor_data in unixt_factor_data.items():
    ls_FRA[factor] = al.performance.factor_rank_autocorrelation(factor_data)

ls_FRA.plot(title="Factor Rank Autocorrelation")

"""
Ratio Sharpe des alphas

"""
def sharpe_ratio(factor_returns, annualization_factor):
    """
    Obtention du ratio sharpe pour chaque facteur pour toute la période

    Paramètres
    ----------
    factor_returns : DataFrame
        Rendements factoriels pour chaque facteur et date
    annualization_factor: float
        Facteur d'annualisation

    Returns
    -------
    sharpe_ratio : Séries de floats pandas
        Ratio Sharpe
    """
    return annualization_factor * factor_returns.mean() / factor_returns.std()


project_tests.test_sharpe_ratio(sharpe_ratio)

daily_annualization_factor = np.sqrt(252)
sharpe_ratio(ls_factor_returns, daily_annualization_factor).round(2)


"""
Nous combinons les facteurs alpha
"""

selected_factors = all_factors.columns[[1, 2, 4]]
print('Selected Factors: {}'.format(', '.join(selected_factors)))

all_factors['alpha_vector'] = all_factors[selected_factors].mean(axis=1)
alphas = all_factors[['alpha_vector']]
alpha_vector = alphas.loc[all_factors.index.get_level_values(0)[-1]]
alpha_vector.head()
