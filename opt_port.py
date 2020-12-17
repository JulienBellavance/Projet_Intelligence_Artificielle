from ev_alp_fac.py import *

"""
Nous disposons d'un modèle alpha et d'un modèle de risque.
Trouvons un portefeuille qui se négocie le plus près possible du modèle alpha
et qui limite le risque tel que mesuré par le modèle de risque.
Nous allons créer l'optimiseur pour ce portefeuille.

"""

from abc import ABC, abstractmethod


class AbstractOptimalHoldings(ABC):
    @abstractmethod
    def _get_obj(self, weights, alpha_vector):
        """
        Obtention de la fonction objectif

        Paramètres
        ----------
        weights : Variable CVXPY
            Poids du Portfolio
        alpha_vector : DataFrame
            Vecteur Alpha

        Returns
        -------
        objective : Objectifs de CVXPY
            Fonction Objectif
        """

        raise NotImplementedError()

    @abstractmethod
    def _get_constraints(self, weights, factor_betas, risk):
        """
        Obtention des contraintes

        Paramètres
        ----------
        weights :Variable CVXPY
            Portfolio weights
        factor_betas : 2D Ndarray
            Facteurs betas
        risk: CVXPY Atom
            Variance prévue des rendements du portefeuille

        Returns
        -------
        constraints : Liste des contraintes CVXPY
            Constraints
        """

        raise NotImplementedError()

    def _get_risk(self, weights, factor_betas, alpha_vector_index, factor_cov_matrix, idiosyncratic_var_vector):
        f = factor_betas.loc[alpha_vector_index].values.T * weights
        X = factor_cov_matrix
        S = np.diag(idiosyncratic_var_vector.loc[alpha_vector_index].values.flatten())

        return cvx.quad_form(f, X) + cvx.quad_form(weights, S)

    def find(self, alpha_vector, factor_betas, factor_cov_matrix, idiosyncratic_var_vector):
        weights = cvx.Variable(len(alpha_vector))
        risk = self._get_risk(weights, factor_betas, alpha_vector.index, factor_cov_matrix, idiosyncratic_var_vector)

        obj = self._get_obj(weights, alpha_vector)
        constraints = self._get_constraints(weights, factor_betas.loc[alpha_vector.index].values, risk)

        prob = cvx.Problem(obj, constraints)
        prob.solve(max_iters=500)

        optimal_weights = np.asarray(weights.value).flatten()

        return pd.DataFrame(data=optimal_weights, index=alpha_vector.index)



"""
En utilisant cette classe comme classe de base, nous implémentons la classe OptimalHoldings.
Il y a deux fonctions qui sont implémentées dans cette classe, les fonctions _get_obj et _get_constraints.

La fonction _get_obj doit renvoyer une fonction objectif CVXPY qui maximise 𝛼𝑇 ∗ 𝑥, où 𝑥 est le poids du portefeuille et 𝛼 est le vecteur alpha.

La fonction _get_constraints doit renvoyer une liste des contraintes suivantes:

𝑟 ≤ 𝑟𝑖𝑠𝑘2cap
𝐵𝑇 ∗ 𝑥⪯𝑓𝑎𝑐𝑡𝑜𝑟max
𝐵𝑇 ∗ 𝑥⪰𝑓𝑎𝑐𝑡𝑜𝑟min
𝑥𝑇𝟙 = 0
‖𝑥‖1≤1
𝑥⪰𝑤𝑒𝑖𝑔ℎ𝑡𝑠min
𝑥⪯𝑤𝑒𝑖𝑔ℎ𝑡𝑠max

Où 𝑥 représente les pondérations du portefeuille, 𝐵 est le facteur bêtas et 𝑟 est le risque du portefeuille
"""

class OptimalHoldings(AbstractOptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Paramètres
        ----------
        weights : Variable CVXPY
            Poids du Portfolio
        alpha_vector : DataFrame
            Vecteur Alpha

        Returns
        -------
        objective : Objectif CVXPY
            Fonction Objectif
        """
        assert(len(alpha_vector.columns) == 1)

        objective = cvx.Maximize(alpha_vector.T.values * weights)

        return objective

    def _get_constraints(self, weights, factor_betas, risk):
        """
        Obtenez les contraintes

        Paramètres
        ----------
        weights : Variable CVXPY
            Poids Portfolio
        factor_betas : 2D Ndarray
            Facteurs Beta
        risk: CVXPY Atom
            Variance prévue des rendements du portefeuille

        Returns
        -------
        constraints : Liste des contraintes CVXPY
            Contraintes
        """
        assert(len(factor_betas.shape) == 2)

        constraints = [risk <= self.risk_cap**2,
                      factor_betas.T * weights <= self.factor_max,
                      factor_betas.T * weights >= self.factor_min,
                      sum(weights) == 0.0,
                      sum(cvx.abs(weights)) <= 1.0,
                      weights >= self.weights_min,
                      weights <= self.weights_max]

        return constraints

    def __init__(self, risk_cap=0.05, factor_max=10.0, factor_min=-10.0, weights_max=0.55, weights_min=-0.55):
        self.risk_cap=risk_cap
        self.factor_max=factor_max
        self.factor_min=factor_min
        self.weights_max=weights_max
        self.weights_min=weights_min


project_tests.test_optimal_holdings_get_obj(OptimalHoldings)
project_tests.test_optimal_holdings_get_constraints(OptimalHoldings)

#Visualisation des poids

optimal_weights = OptimalHoldings().find(alpha_vector, risk_model['factor_betas'], risk_model['factor_cov_matrix'], risk_model['idiosyncratic_var_vector'])

optimal_weights.plot.bar(legend=None, title='Portfolio % Holdings by Stock')
x_axis = plt.axes().get_xaxis()
x_axis.set_visible(False)

project_helper.get_factor_exposures(risk_model['factor_betas'], optimal_weights).plot.bar(
    title='Portfolio Net Factor Exposures',
    legend=False)


"""
Afin de renforcer la diversification, nous utilisons la régularisation dans la fonction objectif.
Nous créons une nouvelle classe appelée OptimalHoldingsRegualization qui obtient ses contraintes de la classe OptimalHoldings.
Dans cette nouvelle classe, nous implémentez la fonction _get_obj pour renvoyer une fonction objectif CVXPY
Qui maximise 𝛼𝑇 ∗ 𝑥 + 𝜆‖𝑥‖2, où 𝑥 est le poids du portefeuille, 𝛼 est le vecteur alpha et 𝜆 est le paramètre de régularisation.

"""

class OptimalHoldingsRegualization(OptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        Obtention de la fonction objectif

        Paramètres
        ----------
        weights : Variable CVXPY
            Poids du Portfolio
        alpha_vector : DataFrame
            Vecteur Alpha

        Returns
        -------
        objective : CVXPY Objective
            Function Objectif
        """
        assert(len(alpha_vector.columns) == 1)
        return cvx.Maximize(alpha_vector.T.values * weights - self.lambda_reg * cvx.norm(weights, 2))

    def __init__(self, lambda_reg=0.5, risk_cap=0.05, factor_max=10.0, factor_min=-10.0, weights_max=0.55, weights_min=-0.55):
        self.lambda_reg = lambda_reg
        self.risk_cap=risk_cap
        self.factor_max=factor_max
        self.factor_min=factor_min
        self.weights_max=weights_max
        self.weights_min=weights_min


project_tests.test_optimal_holdings_regualization_get_obj(OptimalHoldingsRegualization)

#Visualisation des données

optimal_weights_1 = OptimalHoldingsRegualization(lambda_reg=5.0).find(alpha_vector, risk_model['factor_betas'], risk_model['factor_cov_matrix'], risk_model['idiosyncratic_var_vector'])

optimal_weights_1.plot.bar(legend=None, title='Portfolio % Holdings by Stock')
x_axis = plt.axes().get_xaxis()
x_axis.set_visible(False)
project_helper.get_factor_exposures(risk_model['factor_betas'], optimal_weights_1).plot.bar(
    title='Portfolio Net Factor Exposures',
    legend=False)

"""
Optimisation avec un facteur strict de contraintes et une pondération cible

Une autre formulation courante consiste à prendre une pondération cible prédéfinie
𝑥 ∗ (par exemple, un portefeuille quantile), et à résoudre pour se rapprocher au plus près de ce portefeuille
tout en respectant les contraintes au niveau du portefeuille.
Nous allons implémenter la fonction _get_obj pour minimiser sur ‖𝑥 − 𝑥 ∗ ‖2
Où 𝑥 est le poids du portefeuille 𝑥 ∗ est la pondération cible.
"""

class OptimalHoldingsStrictFactor(OptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        obtention de la fonction objectif

        Paramètres
        ----------
        weights : Variable CVXPY
            Poids du Portfolio
        alpha_vector : DataFrame
            Vecteur Alpha

        Returns
        -------
        objective : CVXPY Objectif
            Objective function
        """
        assert(len(alpha_vector.columns) == 1)

        alpha_vector_norm = ((alpha_vector.values - alpha_vector.values.mean()) / np.sum(np.abs(alpha_vector.values))).flatten()
        objective = cvx.Minimize(cvx.norm(weights - alpha_vector_norm))

        return objective


project_tests.test_optimal_holdings_strict_factor_get_obj(OptimalHoldingsStrictFactor)

#Visualisation des données
optimal_weights_2 = OptimalHoldingsStrictFactor(
    weights_max=0.02,
    weights_min=-0.02,
    risk_cap=0.0015,
    factor_max=0.015,
    factor_min=-0.015).find(alpha_vector, risk_model['factor_betas'], risk_model['factor_cov_matrix'], risk_model['idiosyncratic_var_vector'])

optimal_weights_2.plot.bar(legend=None, title='Portfolio % Holdings by Stock')
x_axis = plt.axes().get_xaxis()
x_axis.set_visible(False)

project_helper.get_factor_exposures(risk_model['factor_betas'], optimal_weights_2).plot.bar(
    title='Portfolio Net Factor Exposures',
    legend=False)
