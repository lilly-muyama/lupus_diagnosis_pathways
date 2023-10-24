import pandas as pd
import numpy as np
import argparse
import os
from os.path import isfile, join
import sys
sys.path.append('..')
from modules import utils

def get_steps(filename, prefix):
        try:
            return int(filename[len(prefix)+1:][:-10])
        except Exception as e:
            print(f'Filename: {filename}')
            print(f'Exception: {e}')

def validate_model(folder, X_val, y_val, prefix):
    best_f1, best_acc, best_roc_auc, best_pathway_score, best_wpahm_score = -1, -1, -1, -1, -1
    perf_list = []
    count = 0
    
    for item in os.listdir(folder):        
        if item.startswith(prefix):
            path = join(folder, item)
            if (isfile(path)) & (path.endswith('.zip')):
                count+=1
                if count%10 == 0:
                    print(count)
                model = utils.load_dqn(path)
                pathway_score, wpahm_score, acc, f1, roc_auc, min_length, avg_length, max_length, min_path, max_path = utils.get_val_metrics(model, X_val, y_val)
                perf_dict = {'steps': get_steps(item, prefix), 'pathway_score':pathway_score, 'weighted_pahm_score':wpahm_score, 'acc':acc, 'f1':f1, 
                             'roc_auc':roc_auc, 'min_path_length':min_length, 'avg_length':avg_length, 'max_length':max_length, 'min_path':min_path, 'max_path':max_path} 
                perf_list.append(perf_dict)
                if pathway_score > best_pathway_score:
                    best_pathway_score = pathway_score
                    model.save(f'{folder}/best_pathway_model')
                if wpahm_score > best_wpahm_score:
                    best_wpahm_score = wpahm_score
                    model.save(f'{folder}/best_weighted_pahm_model_9_1')
                if acc > best_acc:
                    best_acc = acc
                    model.save(f'{folder}/best_acc_model')
                if f1 > best_f1:
                    best_f1 = f1
                    model.save(f'{folder}/best_f1_model')
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    model.save(f'{folder}/best_roc_auc_model')

    val_df = pd.DataFrame.from_dict(perf_list) 
    try:
        val_df = val_df.sort_values(by=['steps'])
    except:
        pass
    val_df = val_df.reset_index(drop=True)
    val_df.to_csv(f'{folder}/validation_results.csv', index=False)
    return val_df          

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Validation arguments')
    parser.add_argument('-f', '--folder_path', help='path to folder with model checkpoints to be validated')
    parser.add_argument('-m', '--model_type', help='type of model e.g. dueling_dqn_per')
    args = parser.parse_args()

    print(args)

    val_df = pd.read_csv('../../data/val_set_constant.csv')

    X_val = val_df.iloc[:, 0:-1]
    y_val = val_df.iloc[:, -1]

    X_test, y_test = np.array(X_val), np.array(y_val)

    validate_model(args.folder_path, X_val, y_val, args.model_type)

    