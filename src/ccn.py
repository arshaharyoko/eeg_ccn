import numpy as np
import pywt
from gmm import GMM
from hmm import HMM

import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import json
import math

class CCN:
    """
    Combined Classification Network
    """
    def __init__(self, n_components_gmm, n_components_hmm, n_trials_hmm, wavelet, level, buffer_size=32, random_state=0):
        warnings.filterwarnings("ignore")
        self.n_components_gmm = n_components_gmm
        self.n_components_hmm = n_components_hmm
        self.n_trials_hmm = n_trials_hmm
        self.wavelet = wavelet
        self.level = level
        self.random_state = random_state
        
        self.buffer_size = buffer_size
        self.buffer = None
        self.buffer_preds = None

        self.gmm_models = {k: GMM(n_components=1, random_state=self.random_state) for k in range(n_components_gmm)}
        self.gmm_likelihoods = {k: 0 for k in range(n_components_gmm)}
        self.gmm_labels = {k: 0 for k in range(n_components_gmm)}
        self.hmm = HMM(n_components=self.n_components_hmm, n_trials=self.n_trials_hmm, n_iter=buffer_size, random_state=self.random_state)

    def label_to_onehot(self, label, n_categories):
        onehot = np.zeros(n_categories, dtype=int)
        onehot[label] = 1
        return onehot
    
    def argmax(self, args):
        label = 0
        argmax = 0
        for k, v in args:
            if k==0:
                argmax = v
            if argmax < v:
                argmax = v
                label = k
        return label

    def dwt_extract_features(self, W):
        """
        Parameters:
            W: Input data of length 2^N
        Returns:
            tuple (cAN, cDN)
        """
        coeffs = pywt.wavedec(W, self.wavelet, level=int(math.log2(len(W))), mode="periodization")
        return coeffs[0][0], coeffs[1][0]
    
    def calibrate(self, W_dict, using_preset=False, save_preset=True):
        """
        Parameters:
            W_dict: state dictionary of sequences with len(buffer_size)
        """
        # 1. Figure out maximum length
        features_dict = {}
        if using_preset:
            features_dict = W_dict
        else:
            if save_preset:
                # Save captured signal windows for future preset
                with open(Path(__file__).parent / "features_dict.txt", 'w') as f:
                    json.dump(W_dict, f, indent=2)

        for k, v in W_dict.items():
            features_dict[k] = []
            for W in v:
                cAN, cDN = self.dwt_extract_features(W)
                features_dict[k].append([cAN, cDN])

        # 2. Initialize and fit GMM
        # means = [] 
        # covariances = []
        # weights = []
        # total_samples = sum(len(v) for v in features_dict.values())
        # for k in sorted(features_dict.keys()):
        #     X = np.vstack(features_dict[k])
        #     means.append(np.mean(X, axis=0))
        #     covariances.append(np.cov(X, rowvar=False))
        #     weights.append(X.shape[0]/total_samples)

        for k, gmm in self.gmm_models.items():
            gmm.fit(features_dict[k])

        onehot_sequences = []
        for k, features in features_dict.items():
            for v in features:
                for ki, gmm in self.gmm_models.items():
                    self.gmm_likelihoods[ki] = gmm.likelihood([v])
                onehot_sequences.append(self.label_to_onehot(self.argmax(self.gmm_likelihoods.items()), self.n_components_gmm))

        # DEBUG: Plot the covariance using the window sums of the full training sequence
        # s_dwt_full_sequence = [[idx, v] for idx, v in enumerate([reduce(lambda x,y: x+y, vw_extracted) for k, vw_extracted in features_dict.items()][0])]
        # print(s_dwt_full_sequence[:,0])

        # onehot_sequence = np.array([self.label_to_onehot(label, self.n_components_gmm) for label in gmm_labels])

        # # 3. Initialize and fit HMM
        self.hmm.model.startprob_ = np.ones(self.n_components_hmm)
        self.hmm.model.transmat_ = np.full((self.n_components_hmm, self.n_components_hmm), 0.1) / self.n_components_gmm
        np.fill_diagonal(self.hmm.model.transmat_, 0.9)
        self.hmm.model.emissionprob_ = np.ones((self.n_components_hmm, self.n_components_gmm)) / self.n_components_gmm
        self.hmm.fit(onehot_sequences)

    def process(self, W):
        """
        Parameters:
            X: numpy array of len(buffer_size)
        Returns:
            tuple (label, hidden_states)
        """
        # 1. Preprocess data
        features = np.array(self.dwt_extract_features(W))
        if self.buffer is None or self.buffer.shape[1] != features.shape[0]:
            self.buffer = np.tile(features, (self.buffer_size, 1))
        else:
            self.buffer = np.roll(self.buffer, -1, axis=0)
            self.buffer[-1] = features

        # 2. Classification
        # for k in range(len(features_dict.items())):
        #     self.gmm_likelihoods[k] = self.gmm_models[k]
        for k, gmm in self.gmm_models.items():
            self.gmm_likelihoods[k] = gmm.likelihood(self.buffer) # Log-likelihood score

        label = self.argmax(self.gmm_likelihoods.items())
        if self.buffer_preds is None:
            self.buffer_preds = np.tile(features, (self.buffer_size, 1))
            self.buffer_preds = np.tile([0 for i in range(self.n_components_gmm)], (self.buffer_size, 1))
        else:
            self.buffer_preds = np.roll(self.buffer_preds, -1, axis=0)
            self.buffer_preds[-1] = self.label_to_onehot(label, self.n_components_gmm)

        # gmm_probabilities = self.gmm.probability(self.buffer)
        # gmm_labels = []
        # onehot_sequence = np.array([self.label_to_onehot(label, self.n_components_gmm) for label in gmm_labels])

        # 3. Transitions 
        hidden_states = self.hmm.predict(self.buffer_preds)

        return label, hidden_states