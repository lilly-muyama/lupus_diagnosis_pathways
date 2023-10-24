import pandas as pd
import numpy as np
import sys
sys.path.append('..')
import warnings
from modules import utils
import argparse
warnings.filterwarnings("ignore")



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Performance arguments')
    parser.add_argument('-f', '--model_name', help='filename of the model with extension. It should be in the models folder')
    args = parser.parse_args()

    print(args)

    test_df = pd.read_csv('../../data/test_set_constant.csv')

    X_test = test_df.iloc[:, 0:-1]
    y_test = test_df.iloc[:, -1]

    X_test, y_test = np.array(X_test), np.array(y_test)

    try:
        dqn_model = utils.load_dqn(f'../../models/{args.model_name}')
    except:
        print('This model does not exist. Please use the filename of an existing model')
    dqn_test_df = utils.evaluate_dqn(dqn_model, X_test, y_test)
    dqn_acc, dqn_f1, dqn_roc_auc = utils.test(dqn_test_df['y_actual'], dqn_test_df['y_pred'])
    dqn_pathway_score = utils.get_avg_pathway_score(test_df)
    dqn_wpahm_score = utils.get_weighted_pahm_score([dqn_acc, dqn_pathway_score], [0.9, 0.1])
    print(f'RESULTS FOR {args.model_name}')
    print(f'acc:{dqn_acc}, pathway score:{dqn_pathway_score}, wpahm_score:{dqn_wpahm_score}, f1:{dqn_f1}, roc_auc:{dqn_roc_auc}, mean_episode_length: {dqn_test_df.episode_length.mean()}')
    
    