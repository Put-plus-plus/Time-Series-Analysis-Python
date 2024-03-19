from zipfile import ZipFile 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.callbacks import EarlyStopping   



class Data:

    def __init__(self):

        self.df_raw = None
        self.df_clean = None




    @staticmethod
    def extract_data(zipfile_path, csvfile_name, csvfile_destination):   

        ''' Accepts a string (zipfile_path) with the path including the complete file name of a zip file, a string (csvfile_name) with 
            the name of a compressed csv file contained within the zip file, and a string (csvfile_destination) with the desired
            destination destination of the csv file to be extracted. Outputs an uncompressed csv file.'''  

        try:
            with ZipFile(zipfile_path, 'r') as zf:
                zf.extract(csvfile_name, csvfile_destination)

        except:
            print(f'The {Data.extract_data.__name__} method could not extract in the csv file from the zip file.')




    def read_data(self, csvfile_path):  

        ''' Accepts a string (csvfile_path) with the path including the complete file name of a csv file. Reads a csv file in as a 
            dataframe with raw data (self.df_raw).'''  

        try:
            self.df_raw = pd.read_csv(csvfile_path)

        except:
            print(f'The {Data.read_data.__name__} method could not read in the csv file.')




    def clean_data(self, date_col, event_col, event_min, event_max): 
    
        ''' Accepts a string (date_col) with the name of the date column, a string (event_col) with the name of the column with counts 
        of events, an integer (event_min) with the minimum permitted number of events, and an integer (event_max) with the maximum 
        permitted number of events. Date data must be in the %d/%m/%Y format. A dataframe with raw data (self.df_raw) is filtered to
        produce a dataframe (self.df_clean) without date data entry errors and with permitted counts of events, formatted so that a 
        column with counts of events is the only column and date data is set as the dataframe index.'''

        assert isinstance(self.df_raw, pd.core.frame.DataFrame), f'{Data.clean_data.__name__} error, self.df_raw must be a dataframe' 
        assert len(self.df_raw) > 0, f'{Data.clean_data.__name__} error, self.df_raw can not be an empty dataframe'

        try:
            self.df_raw = self.df_raw[[date_col, event_col]]

            self.df_raw[date_col] = pd.to_datetime(self.df_raw[date_col], format="%d/%m/%Y")
            first_date = self.df_raw[date_col].iloc[0]
            last_date = self.df_raw[date_col].iloc[-1] 
            self.df_raw = self.df_raw.loc[(self.df_raw[date_col] >= first_date) & (self.df_raw[date_col] <= last_date)]

            self.df_raw = self.df_raw[(self.df_raw[event_col] >= event_min) & (self.df_raw[event_col] <= event_max)]
            self.df_raw = self.df_raw.set_index([date_col], drop=True)     

            self.df_clean = self.df_raw.dropna()

        except:
            print(f'The {Data.clean_data.__name__} method could not clean the data.')


    


class NeuralNetwork:

    def __init__(self):

        self.X_train = None
        self.y_train = None 
        self.X_test = None 
        self.y_test = None 
        self.nn_model = None  




    def split_data(self, df_cleaned, split_date):                           
        
        '''Accepts a dataframe (df_cleaned) with the date data as the index and the event data in a single column. Accepts a string (split_date) 
          in the "%d/%m/%Y" format for defining the cutoff for the split of the data into train and test datasets. Output is four numpy arrays with 
          scaled train (X_train, y_train) and test (X_test, y_test) data.''' 
    
        assert isinstance(df_cleaned, pd.core.frame.DataFrame), 'Error, df_cleaned must be a dataframe.' 
        assert len(df_cleaned) > 0, 'Error, df_cleaned can not be an empty dataframe.'

        try:
            split_date = pd.Timestamp(split_date)                             
            df_train = df_cleaned.loc[:split_date]                            
            df_test = df_cleaned.loc[split_date:]                             

            self.scaler = MinMaxScaler(feature_range=(-1, 1))                                           
            array_train_scaled = self.scaler.fit_transform(df_train)                                     
            array_test_scaled = self.scaler.transform(df_test)

            self.X_train = array_train_scaled[:-1]                                
            self.y_train = array_train_scaled[1:]
            self.X_test = array_test_scaled[:-1]
            self.y_test = array_test_scaled[1:]

        except:
            print(f'The {NeuralNetwork.split_data.__name__} could not split the data into numpy arrays with test and train data.')
        


    
    def define_model(self, in_nodes=2):  
        
        '''Accepts an integer (in_nodes) with the number of nodes in the input layer, and with the default set to two. Outputs the 
           structure (self.nn_model) of a recursive neural network model.'''  

        try: 
            out_nodes = 1 
            hidden_nodes = int(round((in_nodes + out_nodes)/2, 0))  

            self.nn_model = Sequential()
            self.nn_model.add(Dense(in_nodes, input_dim=1, activation='relu'))    
            self.nn_model.add(Dense(hidden_nodes))    
            self.nn_model.add(Dense(out_nodes))   
            self.nn_model.compile(loss='mean_squared_error', optimizer='adam')

        except:
            print(f'The {NeuralNetwork.define_model.__name__} could not define the structure of the neural network model.')
     
        

        
    def fit_model(self):  
                
        '''Fits a recursive neural network model (self.nn_model) to the data and returns the mean absolute error (mae) and 
           root mean squarred error (rmse).'''    

        try:
            early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
            self.nn_model.fit(self.X_train, self.y_train, epochs=10, batch_size=1, verbose=1, callbacks=[early_stop], shuffle=False)

            y_predicted = self.nn_model.predict(self.X_test)
            y_predicted_unscaled = self.scaler.inverse_transform(y_predicted)  
            y_test_unscaled = self.scaler.inverse_transform(self.y_test)            

            mae = mean_absolute_error(y_test_unscaled, y_predicted_unscaled)      
            rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_predicted_unscaled))   
            return(f'MAE:{round(mae,1)}, RMSE:{round(rmse, 1)}')       

        except:
            print(f'The {NeuralNetwork.fit_model.__name__} could not fit the neural network model.')
        

