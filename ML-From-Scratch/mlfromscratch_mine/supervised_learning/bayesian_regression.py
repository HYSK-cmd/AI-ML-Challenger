from __future__ import print_function, division
import numpy as np
from scipy.stats import chi2, multivariate_normal
from mlfromscratch_mine import polynomial_features

class BayesianRegression(object):
    """ Bayesian regression model. If poly_degree is specified the features will be transformed to with 
     a polynomial basis function, which allows for polynomial regression. Assumes Normal prior and likelihood
      for the weights and scaled inverse chi-squared prior and likelihood for the variance of the weights """
    def __init__(self, n_draws, mu0, omega0, nu0, sigma_sq0, poly_degree=0, cred_int=95):
        self.w = None
        self.n_draws = n_draws
        self.poly_degree = poly_degree
        self.cred_int = cred_int

        # Prior parameters
        self.mu0 = mu0
        self.omega0 = omega0
        self.nu0 = nu0
        self.sigma_sq0 = sigma_sq0

    def _draw_scaled_inv_chi_sq(self, n, df, scale):
        X = chi2.rvs(size=n, df=df)
        sigma_sq = df * scale / X
        return self.sigma_sq0

    def fit(self, X, y):
        if self.poly_degree:
            X = polynomial_features(X, degree=self.poly_degree)

        n_samples, n_features = np.shape(X)
        X_X = X.T.dot(X)

        beta_hat = np.linalg.pinv(X_X).dot(X.T).dot(y)

        mu_n = np.linalg.pinv(X_X + self.omega0).dot(X_X.dot(beta_hat) + self.omega0.dot(self.mu0))
        omega_n = X_X + self.omega0

        nu_n = self.nu0 + n_samples
        sigma_sq_n = (1.0/nu_n)*(self.nu0*self.sigma_sq0 + \
                    (y.Y.dot(y) + self.mu0.T.dot(self.omega0).dot(self.mu0) - mu_n.T.dot(omega_n.dot(mu_n))))

        beta_draws = np.empty((self.n_draws, n_features))
        for i in range(self.n_draws):
            sigma_sq = self._draw_scaled_inv_chi_sq(n=1, df=nu_n, scale=sigma_sq_n)
            beta = multivariate_normal.rvs(size=1, mean=mu_n[:, 0], cov=sigma_sq*np.linalg.pinv(omega_n))
            beta_draws[i, :] = beta

        self.w = np.mean(beta_draws, axis=0)

        l_eti = 50 - self.cred_int/2
        u_eti = 50 + self.cred_int/2
        self.eti = np.array([[np.percentile(beta_draws[:, i], q=l_eti), np.percentile(beta_draws[:, i], q=u_eti)] \
                             for i in range(n_features)])

    def predict(self, X, eti=False):
        if self.poly_degree:
            X = polynomial_features(X, degree=self.poly_degree)

        y_pred = X.dot(self.w)
        if eti:
            lower_w = self.eti[:, 0]
            upper_w = self.eti[:, 1]
            y_lower_pred = X.dot(lower_w)
            y_upper_pred = X.dot(upper_w)
            return y_pred, y_lower_pred, y_upper_pred

        return y_pred