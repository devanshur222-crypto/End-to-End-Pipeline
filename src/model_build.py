import pandas as pd 
import numpy as np
import logging
import os
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

logdir='log'
os.makedirs(logdir,exist_ok=True)

logger=logging.getLogger('Model_training')
logger.setLevel('DEBUG')

console_log=logging.StreamHandler()
console_log.setLevel('DEBUG')

file_path=os.path.join(logdir,'Model_training.log')
file_handler=logging.FileHandler(file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_log.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_log)
logger.addHandler(file_handler)

def model_train(features : pd.DataFrame , target : pd.Series , params_dt  ):
    try:
        if(features.shape[0] != target.shape[0]):
            raise ValueError('features and target size mismatch.')
        logger.debug('Model training started .')

        dt_model=DecisionTreeRegressor(random_state=42)
        dt_model.fit(features,target)

        logger.debug('Model training done successfully.')
        return dt_model
    except Exception as e:
        logger.error('Unexpected error occured at model train .')
        raise

def model_save(model,file_path):
    try:
        logger.debug('Model saving initiated.')
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as file :
            pickle.dump(model,file)
        logger.debug('Model saving completed.')
    except Exception as e:
        logger.error('Unexpected error caused during model saving.')
   


def main():
    try:
        logger.debug('Main function started.')
        params_dt={'random_state':42}
        train_data=pd.read_csv(r'C:\Users\devan\OneDrive\Desktop\End-to-End-Pipeline\data\interim\train_data.csv')
        target=train_data['selling_price']
        train_data.drop(columns='selling_price',inplace=True)
        features=train_data

        dt_model=model_train(features,target,params_dt)

        file_path='models/model.pkl'
        model_save(dt_model,file_path)
        logger.debug('Main function ended. Model training done successfully.')
    except Exception as e:
        logger.error('Unable to do model training.')

if __name__ == '__main__' :
    main()



    
    

