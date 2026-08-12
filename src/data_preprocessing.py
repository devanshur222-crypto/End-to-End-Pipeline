import pandas as pd
import numpy as np
import logging 
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import yaml

encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')

logdir='log'
os.makedirs(logdir,exist_ok=True)
logger=logging.getLogger('Data_preprocessing')
logger.setLevel('DEBUG')

console_logger=logging.StreamHandler()
console_logger.setLevel('DEBUG')


log_file_path=os.path.join(logdir,'Data_preprocessing.log')
fil_log=logging.FileHandler(log_file_path)
fil_log.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_logger.setFormatter(formatter)
fil_log.setFormatter(formatter)

logger.addHandler(console_logger)
logger.addHandler(fil_log)

def load_params(par_path):
    try:
        with open(par_path,'r') as file:
            params=yaml.safe_load(file)
        return params
        logger.debug('Parameters loaded successfully.')
    except FileNotFoundError:
        logger.error('File not found')
        raise
    except yaml.YAMLError as e:
        logger.error('Unable to load yaml.')
        raise
    except Exception as e:
        logger.error('Unexpected error occured during loading of parameters.')
        raise

def transform(df,categorical_cols):
    try:
        logger.debug('Transformation started.')
        transformed=encoder.fit_transform(df[categorical_cols])
        df.drop(columns=categorical_cols,inplace=True)
        transformed=pd.DataFrame(transformed)
        df=pd.concat([df,transformed],axis=1)
        logger.debug('Transformation completed')
        return df
    except Exception as e :
        logger.error('Unexpected error caused in tranform.')
        raise

def main():
    try:
        logger.debug('Main function started.')
        raw_df=pd.read_csv(r'C:\Users\devan\OneDrive\Desktop\End-to-End-Pipeline\data\raw\raw_data')
        categorical_cols=['brand','fuel','owner']
        df=transform(raw_df,categorical_cols)
        params=load_params(par_path='params.yaml')
        test_size=params['data_preprocessing']['test_size']
        print(test_size)
        train_data, test_data = train_test_split(df,test_size=test_size,random_state=42)
        data_path=os.path.join('./data','interim')
        os.makedirs(data_path,exist_ok=True)
        train_data.to_csv(os.path.join(data_path,'train_data.csv'))
        test_data.to_csv(os.path.join(data_path,'test_data.csv'))
        logger.debug('Main function executed successfully.')
    except Exception as e:
        logger.error('Unable to preprocess the data.')
        raise

if __name__ == '__main__' :
    main()




        


    
