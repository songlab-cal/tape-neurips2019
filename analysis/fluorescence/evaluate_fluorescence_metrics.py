from typing import List

import numpy as np 
import pickle as pkl
import pandas as pd
import scipy
from sklearn.metrics import mean_squared_error 
from scipy.stats import spearmanr
                                      

MODELS = [
        'bepler', 
        'unirep', 
        'transformer',
        'transformer_pretrain',
        'lstm',
        'lstm_pretrain',
        'resnet',
        'resnet_pretrain',
        'one_hot'
]

METRICS = [
        'mse', 
        'spearmanr',
        'bright_mse',
        'bright_spearmanr',
        'dark_mse',
        'dark_spearmanr'
]

BRIGHT_CUTOFF = 2.0

def postprocess(data: List[np.ndarray]) -> np.ndarray:
    """ Converts list of numpy arrays to flat numpy array. 
    """
    _clean = [float(i) for i in data]
    clean = np.array(_clean)
    return clean

def load_results(model: str):
        """Load all predictions for a given model. 

        Returns 
                results: numpy array of dim [n_val_set, max_length, 2] 
                 last dimension has [prediction]
        """
        model_results_path = model + '_outputs.pkl'
        with open(model_results_path, 'rb') as f: 
                results = pkl.load(f) 
        sequences = results['primary']
        predictions = postprocess(results['prediction'])
        true_values = postprocess(results['log_fluorescence'])
        num_mutations = postprocess(results['num_mutations'])

        return sequences, predictions, true_values, num_mutations

if __name__ == '__main__':

        # Create a dataframe with all results  
        full_results = pd.DataFrame(0, index=MODELS, columns=METRICS)

        # Grab metrics for each model 
        for model in MODELS: 
                # Compute metrics for full test data
                sequences, predictions, true_values, num_mutations = load_results(model)
                mse = mean_squared_error(true_values, predictions)
                spearman_r, _ = spearmanr(true_values, predictions)
                
                full_results.loc[model, 'mse'] = np.mean(mse) 
                full_results.loc[model, 'spearmanr'] = spearman_r

                # Compute metrics for bright mode
                bright_idx = np.where(true_values > BRIGHT_CUTOFF)
                bright_preds = predictions[bright_idx]
                bright_trues = true_values[bright_idx]

                bright_mse = mean_squared_error(bright_trues, bright_preds)
                bright_spearman_r, _ = spearmanr(bright_trues, bright_preds) 

                full_results.loc[model, 'bright_mse'] = np.mean(bright_mse) 
                full_results.loc[model, 'bright_spearmanr'] = bright_spearman_r

                # Compute metrics for dark mode
                dark_idx = np.where(true_values <= BRIGHT_CUTOFF)
                dark_preds = predictions[dark_idx]
                dark_trues = true_values[dark_idx]

                dark_mse = mean_squared_error(dark_trues, dark_preds)
                dark_spearman_r, _ = spearmanr(dark_trues, dark_preds) 

                full_results.loc[model, 'dark_mse'] = np.mean(dark_mse) 
                full_results.loc[model, 'dark_spearmanr'] = dark_spearman_r
        
                # Could do fancier stuff like produce plots based on the full mae, mse, r2

        # Dump whole table to a csv  
        full_results.round(2).to_csv('fluorescence_result_metrics.csv')  
