import numpy as np
from sklearn.mixture import GaussianMixture

class GMM:
    def __init__(self, n_components, random_state=0):
        self.n_components = n_components
        self.random_state = random_state
        self.model = GaussianMixture(n_components=self.n_components, random_state=self.random_state)
    
    def fit(self, X):
        """
        Fit the Gaussian Mixture Model to the feature matrix
        Parameters:
            X: numpy array of shape (n_samples, n_features)
        """
        self.model.fit(X)
    
    def predict(self, X):
        """
        Predict the labels for wave sequence 
        Returns:
            labels: numpy array of shape (n_samples,)
        """
        return self.model.predict(X)
    
    def probability(self, X):
        """
        Returns the probabilities of each component of wave sequence
        """
        return self.model.predict_proba(X)
    
    def likelihood(self, X):
        """
        Returns the per sample average log-likelihood
        """
        return self.model.score(X)
