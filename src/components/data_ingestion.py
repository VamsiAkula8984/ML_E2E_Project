import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#configures where train, test data is saved
@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):# to read data from databases
        logging.info("Data Ingestion started...")
        try:
            #reading data from DB or any other source
            df = pd.read_csv('notebook\\data\\stud.csv')
            logging.info('Dataset reading as dataframe successful!!')

            #creating artifacts directory to save train, test and raw data.csv files
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            #storing raw data in raw.csv
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info(f"Raw data stored at {self.ingestion_config.raw_data_path}")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            logging.info("Train test split initiated")

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info(f"Train data stored at {self.ingestion_config.train_data_path}")

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info(f"Test data stored at {self.ingestion_config.test_data_path}")

            logging.info("Data Ingestion ended:)")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()