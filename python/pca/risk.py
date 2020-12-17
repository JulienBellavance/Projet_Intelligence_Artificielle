from bundler import *
from sklearn.decomposition import PCA


def fit_pca(returns, num_factor_exposures, svd_solver):
    """
    Entraine le modèle PCA model avec les retours.

    Paramètres
    ----------
    retours : DataFrame
        Retours pour chaque symbole de ticker et pour les dates
    num_factor_exposures : int
        Nombre de facteurs pour le PCA
    svd_solver: str
        Le solveur à utiliser pour le modèle PCA

    Retours
    -------
    pca : PCA
        Modèle entrainé sur les retours
    """
    pca = PCA(n_components=num_factor_exposures, svd_solver=svd_solver).fit(returns)
    return pca

project_tests.test_fit_pca(fit_pca)


"""
Visualisation du modèle
"""

#Regardons les composantes du pca
num_factor_exposures = 20
pca = fit_pca(five_year_returns, num_factor_exposures, 'full')

pca.components_

#Variance du pca
plt.bar(np.arange(num_factor_exposures), pca.explained_variance_ratio_)

"""
Implémentation des facteurs betas à partir du modèle PCA
"""
def factor_betas(pca, factor_beta_indices, factor_beta_columns):
    """
    Paramètres
    ----------
    pca : PCA
        Modèle entrainé sur les retours
    factor_beta_indices : 1D Ndarray
        Indices des facteurs beta
    factor_beta_columns : 1D Ndarray
        Colomnes des facteurs

    Returns
    -------
    factor_betas : DataFrame
        Facteur betas
    """
    assert len(factor_beta_indices.shape) == 1
    assert len(factor_beta_columns.shape) == 1

    factor_betas = pd.DataFrame(pca.components_.T, factor_beta_indices, factor_beta_columns)

    return factor_betas


project_tests.test_factor_betas(factor_betas)

#Visualisation des facteurs bêtas
risk_model = {}
risk_model['factor_betas'] = factor_betas(pca, five_year_returns.columns.values, np.arange(num_factor_exposures))

risk_model['factor_betas']


"""
Obtention des retours factoriels du modèle PCA à l'aide des données de retour.
"""

def factor_returns(pca, returns, factor_return_indices, factor_return_columns):
    """
    Paramètres
    ----------
    pca : PCA
        Modèle entrainé sur les retours
    returns : DataFrame
        Retours pour chaque code ticker et pour chaque date
    factor_return_indices : 1D Ndarray
        Indices Facteurs de retours
    factor_return_columns : 1D Ndarray
        Colonnes des facteurs de retour

    Retours
    -------
    factor_returns : DataFrame
        Facteurs de retour
    """
    assert len(factor_return_indices.shape) == 1
    assert len(factor_return_columns.shape) == 1

    factor_returns = pd.DataFrame(pca.transform(returns), factor_return_indices, factor_return_columns)

    return factor_returns


project_tests.test_factor_returns(factor_returns)

#Visualisation des facteurs de retour
risk_model['factor_returns'] = factor_returns(
    pca,
    five_year_returns,
    five_year_returns.index,
    np.arange(num_factor_exposures))

risk_model['factor_returns'].cumsum().plot(legend=None)


"""
Obtention la matrice de covariance des facteurs.
"""

def factor_cov_matrix(factor_returns, ann_factor):
    """
    Paramètres
    ----------
    factor_returns : DataFrame
        Retours sur les facteurs
    ann_factor : int
        Facteur d'annualisation

    Retours
    -------
    factor_cov_matrix : 2D Ndarray
        Matrice de facteur de covariance
    """
    return np.diag(factor_returns.var(axis=0, ddof=1)*ann_factor)

project_tests.test_factor_cov_matrix(factor_cov_matrix)

#Visualisation des facteurs de la matrice de covariance
ann_factor = 252
risk_model['factor_cov_matrix'] = factor_cov_matrix(risk_model['factor_returns'], ann_factor)

risk_model['factor_cov_matrix']

"""
Obtention de la matrice de variance idiosyncratique.
"""

def idiosyncratic_var_matrix(returns, factor_returns, factor_betas, ann_factor):
    """
    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    factor_returns : DataFrame
        Factor returns
    factor_betas : DataFrame
        Factor betas
    ann_factor : int
        Annualization factor

    Returns
    -------
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix
    """

    common_returns = pd.DataFrame(np.dot(factor_returns, factor_betas.T), returns.index, returns.columns)
    residuals = (returns - common_returns)
    idiosyncratic_var_matrix = pd.DataFrame(np.diag(np.var(residuals))*ann_factor, returns.columns, returns.columns)

    return idiosyncratic_var_matrix


project_tests.test_idiosyncratic_var_matrix(idiosyncratic_var_matrix)

risk_model['idiosyncratic_var_matrix'] = idiosyncratic_var_matrix(five_year_returns, risk_model['factor_returns'], risk_model['factor_betas'], ann_factor)

risk_model['idiosyncratic_var_matrix']

"""
Obtention du vecteur de variance idiosyncratique.
"""

def idiosyncratic_var_vector(returns, idiosyncratic_var_matrix):
    """
    Paramètres
    ----------
    returns : DataFrame
        Retours pour chaque symbole ticker et chaque date
    idiosyncratic_var_matrix : DataFrame
        Matrice de variance idiosyncratique

    Returns
    -------
    idiosyncratic_var_vector : DataFrame
        Vecteur de variance idiosyncratique
    """
    return pd.DataFrame(np.array(idiosyncratic_var_matrix).diagonal(), index = idiosyncratic_var_matrix.index)

project_tests.test_idiosyncratic_var_vector(idiosyncratic_var_vector)

#Visualisation vecteur de variance idiosyncratique
risk_model['idiosyncratic_var_vector'] = idiosyncratic_var_vector(five_year_returns, risk_model['idiosyncratic_var_matrix'])

risk_model['idiosyncratic_var_vector']



"""
Prédiction du risque du portfolio en utilisant la formule  √𝑋𝑇(𝐵𝐹𝐵𝑇+𝑆)𝑋
X: Poids du Portfolio
B: Facteurs bêtas
F: Matrices de facteurs de covariance
S: Matrice de variance idiosyncratique

"""
def predict_portfolio_risk(factor_betas, factor_cov_matrix, idiosyncratic_var_matrix, weights):
    """
    Obtention du risque de portfolio

    La formule du risque de portefeuille prévu est sqrt(X.T(BFB.T + S)X) où:
      X est le poids du portefeuille
      B est l'ensemble des facteurs beta
      F est le facteur de covariance de la matrice
      S est la varianc idiosyncratique de la matrice

    Paramètres
    ----------
    factor_betas : DataFrame
        Facteurs betas
    factor_cov_matrix : 2 dimensional Ndarray
        Facteurs de covariance de la matrice
    idiosyncratic_var_matrix : DataFrame
        Variance idiosyncratique de la matrice
    weights : DataFrame
        Poids des Portfolio

    Returns
    -------
    predicted_portfolio_risk : float
        Risque prédit du portfolio
    """
    assert len(factor_cov_matrix.shape) == 2

    predict_portfolio_risk = np.sqrt(weights.T.dot(factor_betas@factor_cov_matrix@factor_betas.T+idiosyncratic_var_matrix).dot(weights))
    return predict_portfolio_risk[0][0]


project_tests.test_predict_portfolio_risk(predict_portfolio_risk)

#Visualisation du portfolio
all_weights = pd.DataFrame(np.repeat(1/len(universe_tickers), len(universe_tickers)), universe_tickers)

predict_portfolio_risk(
    risk_model['factor_betas'],
    risk_model['factor_cov_matrix'],
    risk_model['idiosyncratic_var_matrix'],
    all_weights)
