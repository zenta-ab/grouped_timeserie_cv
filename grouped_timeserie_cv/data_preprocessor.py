import pandas as pd


class DataPreprocessor:
    @staticmethod
    def preprocess_data(data, frequency ='D', datetime_column ='DateTime', label_column = 'Label'):
        data_copy =  data.copy()
        data_copy[datetime_column] = pd.to_datetime(data_copy[datetime_column])
        data_copy.set_index(datetime_column, inplace=True)
        groups =   data_copy.index.to_period(frequency).astype(str)
        X = data.drop(columns=[label_column,  datetime_column])
        y = data[label_column]
    
        return X, y, groups