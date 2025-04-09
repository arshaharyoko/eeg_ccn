import numpy as np
from hmmlearn.hmm import MultinomialHMM

class HMM:
    def __init__(self, n_components, n_trials, n_iter, random_state=0):
        self.n_components = n_components
        self.n_trials = n_trials
        self.random_state = random_state
        self.model = MultinomialHMM(n_components=self.n_components, n_trials=self.n_trials, n_iter=n_iter, startprob_prior=1.0, transmat_prior=1.0, random_state=self.random_state)
    
    def fit(self, X):
        """
        Fit HMM on the observation sequence
        Parameters:
            X: numpy array of shape (n_samples, 1)
        """
        self.model.fit(X)
    
    def predict(self, X):
        """
        Predict the most likely hidden state given the observation sequence
        Returns:
            hidden_states: numpy array of shape (n_samples,)
        """
        return self.model.predict(X)
    
    def probability(self, X):
        """
        Log probability of the observation sequence
        """
        return self.model.score(X)
