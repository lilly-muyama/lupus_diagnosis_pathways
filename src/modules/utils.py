
import ast
import os
import random
import numpy as np
import pandas as pd
from modules.env import LupusEnv
from modules.constants import constants
import tensorflow as tf
from stable_baselines import DQN
from stable_baselines.common.callbacks import CheckpointCallback
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

random.seed(constants.SEED)
np.random.seed(constants.SEED)
os.environ['PYTHONHASHSEED']=str(constants.SEED)
tf.set_random_seed(constants.SEED)
tf.compat.v1.set_random_seed(constants.SEED)


def get_pathway_features_score(pathway):
    ''''
    Computes the total feature score of a single pathway
    '''
    if isinstance(pathway, str):
        pathway = ast.literal_eval(pathway)
    total_feature_cost = 0
    for feat in pathway[:-1]:
        feat_score = constants.FEATURE_SCORES[feat]
        total_feature_cost += feat_score
    return total_feature_cost

def get_avg_pathway_score(test_df):
    '''
    Computes the average pathway score of a model
    '''
    total_score = 0
    for i, row in test_df.iterrows():
        row_feature_score = get_pathway_features_score(row.trajectory)
        total_row_score = 1-(row_feature_score/constants.MAX_FEATURE_SCORE)
        total_score += total_row_score
        
    return (total_score/len(test_df))*100

def get_weighted_pahm_score(numbers, weights): #both are list of the same length
    whm = (weights[0] + weights[1]) / ((weights[0] / numbers[0]) + (weights[1] / numbers[1]))
    return whm

def load_dqn(filename, env=None):
    '''
    Loads a previously saved DQN model
    '''
    model = DQN.load(filename, env=env)
    return model

def create_env(X, y, random=True):
    '''
    Creates and environment using the given data
    '''
    env = LupusEnv(X, y, random)
    print(f'The environment seed is {env.seed()}') #to delete
    return env

def stable_vanilla_dqn(X_train, y_train, timesteps, save=False, log_path=None, log_prefix='dqn', filename=None, per=False):
    '''
    Creates and trains a standard DQN model
    '''
    if per:
        log_prefix = 'dqn_per'
    training_env = create_env(X_train, y_train)
    model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, learning_rate=0.0001, buffer_size=1000000, learning_starts=50000, 
                train_freq=4, target_network_update_freq=10000, exploration_final_eps=0.05, n_cpu_tf_sess=1, policy_kwargs=dict(dueling=False),
                double_q=False, prioritized_replay=per)
    
    checkpoint_callback = CheckpointCallback(save_freq=constants.CHECKPOINT_FREQ, save_path=log_path, name_prefix=log_prefix)
    model.learn(total_timesteps=timesteps, log_interval=100000, callback=checkpoint_callback)
    if save:
        model.save(f'{log_path}/{filename}.pkl')
    training_env.close()
    return model

def stable_dueling_dqn(X_train, y_train, timesteps, save=False, log_path=None, log_prefix='dueling_dqn', filename=None, per=False):
    '''
    Creates and trains a dueling DQN model
    '''
    print(F'Using seed {constants.SEED}')
    if per:
        log_prefix = 'dueling_dqn_per'
    training_env = create_env(X_train, y_train)
    model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, learning_rate=0.0001, buffer_size=1000000, learning_starts=50000, 
                train_freq=4, target_network_update_freq=10000, exploration_final_eps=0.05, n_cpu_tf_sess=1, double_q=False, prioritized_replay=per)
    
    checkpoint_callback = CheckpointCallback(save_freq=constants.CHECKPOINT_FREQ, save_path=log_path, name_prefix=log_prefix)
    model.learn(total_timesteps=timesteps, log_interval=100000, callback=checkpoint_callback)
    if save:
        model.save(f'{log_path}/{filename}.pkl')
    training_env.close()
    return model

def stable_dueling_ddqn(X_train, y_train, timesteps, save=False, log_path=None, log_prefix='dueling_ddqn', filename=None, per=False):
    '''
    Creates and trains a dueling double DQN model
    '''
    if per:
        log_prefix = 'dueling_ddqn_per'
    training_env = create_env(X_train, y_train)
    model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, learning_rate=0.0001, buffer_size=1000000, learning_starts=50000, 
                train_freq=4, target_network_update_freq=10000, exploration_final_eps=0.05, n_cpu_tf_sess=1, prioritized_replay=per)
    
    checkpoint_callback = CheckpointCallback(save_freq=constants.CHECKPOINT_FREQ, save_path=log_path, name_prefix=log_prefix)
    model.learn(total_timesteps=timesteps, log_interval=100000, callback=checkpoint_callback)
    if save:
        model.save(f'{log_path}/{filename}.pkl')
    training_env.close()
    return model

def stable_double_dqn(X_train, y_train, timesteps, save=False, log_path=None, log_prefix ='ddqn', filename=None, per=False):
    '''
    Creates and trains a double DQN model
    '''
    if per:
        log_prefix = 'ddqn_per'
    training_env = create_env(X_train, y_train)
    model = DQN('MlpPolicy', training_env, verbose=1, seed=constants.SEED, learning_rate=0.0001, buffer_size=1000000, learning_starts=50000, 
                train_freq=4, target_network_update_freq=10000, exploration_final_eps=0.05, n_cpu_tf_sess=1, policy_kwargs=dict(dueling=False), 
                prioritized_replay=per)
    
    checkpoint_callback = CheckpointCallback(save_freq=constants.CHECKPOINT_FREQ, save_path=log_path, name_prefix=log_prefix)
    model.learn(total_timesteps=timesteps, log_interval=100000, callback=checkpoint_callback)
    if save:
        model.save(f'{log_path}/{filename}.pkl')
    training_env.close()
    return model

def multiclass(actual_class, pred_class, average = 'macro'):
    '''
    Returns the ROC-AUC score for multi-labeled data
    '''

    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        other_class = [x for x in unique_class if x != per_class]
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc
    avg = sum(roc_auc_dict.values()) / len(roc_auc_dict)
    return avg

def test(ytest, ypred):
    '''
    Return performance metrics for a model
    '''
    acc = accuracy_score(ytest, ypred)
    f1 = f1_score(ytest, ypred, average ='macro', labels=np.unique(ytest))
    try:
        roc_auc = multiclass(ytest, ypred)
    except:
        roc_auc = None
    return acc*100, f1*100, roc_auc*100

def evaluate_dqn(dqn_model, X_test, y_test):
    '''
    Evaluates a DQN model on test data
    '''
    test_df = pd.DataFrame()
    env = create_env(X_test, y_test, random=False)
    count=0

    try:
        while True:
            count+=1
            obs, done = env.reset(), False
            while not done:
                action, _states = dqn_model.predict(obs, deterministic=True)
                obs, rew, done, info = env.step(action)
                if done == True:
                    test_df = test_df.append(info, ignore_index=True)
    except StopIteration:
        pass
    return test_df

def get_val_metrics(model, X_val, y_val):
    val_df = evaluate_dqn(model, X_val, y_val)
    acc, f1, roc_auc, = test(val_df['y_actual'], val_df['y_pred'])
    pathway_score = get_avg_pathway_score(val_df)
    wpahm_score = get_weighted_pahm_score([acc, pathway_score], [0.9, 0.1])
    min_path_length = val_df.episode_length.min()
    average_path_length = val_df.episode_length.mean()
    max_path_length = val_df.episode_length.max()
    min_sample_pathway = val_df[val_df.episode_length==min_path_length].trajectory.iloc[0]
    max_sample_pathway = val_df[val_df.episode_length==max_path_length].trajectory.iloc[0]
    return pathway_score, wpahm_score, acc, f1, roc_auc, min_path_length, average_path_length, max_path_length, min_sample_pathway, max_sample_pathway