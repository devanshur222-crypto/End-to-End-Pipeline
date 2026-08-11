import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import logging

log_dir='log' # making a directory name log to save the log information there 
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('Data_ingestion')  # making a logger object and naming it Data ingenstion with level 'DEBUG'
logger.setLevel('DEBUG')

console_log=logging.StreamHandler()
console_log.setLevel('DEBUG')


log_file_path=os.path.join(log_dir,'Data_ingestion.log')
fil_log=logging.FileHandler(log_file_path)
fil_log.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_log.setFormatter(formatter)
fil_log.setFormatter(formatter)

logger.addHandler(console_log)
logger.addHandler(fil_log)


def load_data(path_):
    try :
        df=pd.read_csv(path_)
        logger.debug('Data loading is successfull')
        return df
    except Exception as e :
        logger.error('Unexpected error occured while loading the data')
        raise


def pre_processing(df):
    try :
        df.drop_duplicates(inplace=True)
        logger.debug('Data preprocessing is successfull')
        return df
    except Exception as e :
        logger.error("Unexpected error occured while preprocessing the data")
        raise

def save_data(df : pd.DataFrame , data_path : str ) -> None :
    try :
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        df.to_csv(os.path.join(raw_data_path,'raw_data'), index=False)
        logger.debug('Data have been saved successfully')
    except Exception as e:
        logger.error("Unexpected error occured while saving the data ")
        raise

def main():
    try :
        path_=r'C:\Users\devan\OneDrive\Desktop\End-to-End-Pipeline\Used_Car_Prices.csv'
        df=load_data(path_)
        df=pre_processing(df)
        save_data(df,data_path='./data')
    except Exception as e :
        logger.error("Unable to do the data ingestion process .")
        raise

if __name__ == '__main__' :
    main()

    
