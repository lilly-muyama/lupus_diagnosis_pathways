import pandas as pd
import numpy as np
import random 
import os
import sys
sys.path.append('..')
import tensorflow as tf
from modules.constants import constants
import argparse
from multiprocessing import Process



def run_dqn_model(model_type, seed, steps, lambda_num, per, folder_name):
    # dir_name = f'seed_{seed}_{steps}'
    path = f'../../models/{folder_name}'
    os.mkdir(path)
  
    if model_type =='dqn':
        model = utils.stable_vanilla_dqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='dqn_per', filename=f'dqn_per_{steps}', per=per)
    elif model_type == 'ddqn':
        model = utils.stable_double_dqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='ddqn_per', filename=f'ddqn_per_{steps}', per=per)
    elif model_type == 'dueling_dqn':
        model = utils.stable_dueling_dqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='dueling_dqn_per', filename=f'dueling_dqn_per_{steps}', per=per)
    elif model_type == 'dueling_ddqn':
        model = utils.stable_dueling_ddqn(X_train, y_train, steps, save=True, log_path=path, log_prefix='dueling_ddqn_per', filename=f'dueling_ddqn_per_{steps}', per=per)
    else:
        raise ValueError('Unknown model type!')
    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Parameters for dqn model')
    parser.add_argument('-f', '--train_set_name', help='Name of the training dataset e.g. train_set_basic', default = 'train_set_basic')
    parser.add_argument('-m', '--model_type', help='dqn or ddqn or dueling_dqn or dueling_ddqn', default='dueling_dqn')
    parser.add_argument('-p', '--prioritized_replay', help ='yes or no', default='yes')  
    parser.add_argument('-t', '--steps', help='number of timesteps to train the model', type=int, default=int(10e7))
    parser.add_argument('-s', '--seed', help='seed to use in experiment', type=int, default=42)
    parser.add_argument('-l', '--lambda_constant', help = 'lambda constant in reward function', type=int, default=9)
    args = parser.parse_args()
    constants.init(args)

    from modules import utils

    SEED = constants.SEED
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED']=str(SEED)
    tf.set_random_seed(SEED)
    tf.compat.v1.set_random_seed(SEED)

    print(f'Seed being used: {SEED}')
    print(f'Number of steps: {args.steps}')
    print(f'Lambda being used: {constants.LAMBDA}')


    train_df = pd.read_csv(f'../../data/{args.train_set_name}.csv')
    train_df = train_df.fillna(-1)

    X_train = train_df.iloc[:, 0:-1]
    y_train = train_df.iloc[:, -1]
    X_train, y_train = np.array(X_train), np.array(y_train)   

    if args.prioritized_replay == 'yes':
        per_str= '_per'
    else:
        per_str=''


    folder_name =f'{args.model_type}{per_str}_{args.train_set_name[len("train_set_"):]}_lambda_{args.lambda_constant}_{SEED}_{args.steps}' 
    

    if args.steps >1000000:
        print(f'Training {args.model_type} model over {args.steps} steps. This may take a while')
    else:
        print(f'Training {args.model_type} model over {args.steps} steps')

    run_dqn_model(args.model_type, SEED, args.steps, args.lambda_constant, args.prioritized_replay, folder_name)

    
   