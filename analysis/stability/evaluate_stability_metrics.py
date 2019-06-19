from typing import List

import numpy as np 
import pickle as pkl
import pandas as pd
import scipy
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from scipy.stats import spearmanr
                                      

MODELS = [
        'bepler', 
        'unirep', 
        'transformer',
        'lstm',
        'resnet',
        'transformer_pretrain',
        'lstm_pretrain',
        'resnet_pretrain',
        'one_hot'
]

METRICS = [
        'mse', 
        'spearmanr',
        'precision',
        'recall'
]

TOPOLOGIES = [
        b'HHH',
        b'HEEH',
        b'EHEE',
        b'EEHEE'
]

column_names = [topology.decode() + '_' + metric for metric,topology in zip(METRICS, TOPOLOGIES + ['full'])]

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
        predictions = results['prediction']
        true_values = results['stability_score']
        topologies = results['topology']
        sequence_ids = results['id']
        parents = [seq_id.split(b'.')[0] for seq_id in sequence_ids]

        # Compute predicted and true class for each and return
        return sequences, np.array(predictions), np.array(true_values), topologies, sequence_ids, parents 

if __name__ == '__main__':

        # Create a dataframe with all results  
        full_results = pd.DataFrame(0, index=MODELS, columns=column_names)

        # Grab metrics for each model 
        for model in MODELS: 
                # Compute qualitative metrics 
                sequences, predictions, true_values, topologies, sequence_ids, parents = load_results(model)
                mse = mean_squared_error(true_values, predictions) 
                spearman_r, _ = spearmanr(true_values, predictions)

                full_results.loc[model, 'full_mse'] = np.mean(mse)
                full_results.loc[model, 'full_spearmanr'] = spearman_r
                for topology in TOPOLOGIES:
                    topology_idx = [t == topology for t in topologies]
                    cur_values = true_values[topology_idx]
                    pred_values = predictions[topology_idx]
                    mse = mean_squared_error(cur_values, pred_values)
                    spearman_r, _ = spearmanr(cur_values, pred_values)
                    
                    full_results.loc[model, topology.decode() + '_mse'] = np.mean(mse) 
                    full_results.loc[model, topology.decode() + '_spearmanr'] = spearman_r 

                    # Compute qualitative metrics
                    precision = precision_score(true_classes, predicted_classes)
                    recall = recall_score(true_classes, predicted_classes)

                    full_results.loc[model, topology.decode() + '_precision'] = np.mean(precision)
                    full_results.loc[model, topology.decode() + '_recall'] = np,mean(recall)
                
        # Dump whole table to a csv  
        full_results.round(2).to_csv('stability_result_metrics.csv')  
