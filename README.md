## Description 

In time series analysis a time series, ie a series of data pints indexed in time order, is analysed in order to extract meaningful characteristics of the data. This time series analysis is organised into two classes, `Data` that holds the pre-processing of the data and `NeuralNetwork` that holds the recursive neural network model. 

There are two classes in `timeseriesanalysis.py`, `. The `Data` class has three methods which pre-process the data. The `extract_data()` method accepts a string with the path including the complete file name of a zip file, a string (csvfile_name) with the name of a compressed csv file contained within the zip file, and a string (csvfile_destination) with the desired destination destination of the csv file to be extracted. Outputs an uncompressed csv file. The `read_data()` method accepts a string (csvfile_path) with the path including the complete file name of a csv file. Reads a csv file in as a dataframe with raw data (self.df_raw). The `clean_data()` method accepts a string (date_col) with the name of the date column, a string (event_col) with the name of the column with counts of events, an integer (event_min) with the minimum permitted number of events, and an integer (event_max) with the maximum permitted number of events. Date data must be in the %d/%m/%Y format. A dataframe with raw data (self.df_raw) is filtered to produce a dataframe (self.df_clean) without date data entry errors and with permitted counts of events, formatted so that a column with counts of events is the only column and date data is set as the dataframe index. The `NeuralNetwork`class has three methods which build a a recursive neural network model for time series analysis. The `split_data()` method accepts a dataframe (df_cleaned) with the date data as the index and the event data in a single column. Accepts a string (split_date) in the "%d/%m/%Y" format for defining the cutoff for the split of the data into train and test datasets. Output is four numpy arrays with scaled train (X_train, y_train) and test (X_test, y_test) data. The `define_model()` accepts an integer (in_nodes) with the number of nodes in the input layer, and with the default set to two. Outputs the structure (self.nn_model) of a recursive neural network model. The `fit_model()` method fits a recursive neural network model (self.nn_model) to the data and returns the mean absolute error (mae) and root mean squarred error (rmse).

The dataset used in the example below is an adapted version of Hungarian Chickenpox Cases. (2021). UCI Machine Learning Repository. https://doi.org/10.24432/C5103B, and is used under a Creative Commons Attribution 4.0 International licence. 
  
## Dependencies
* Microsoft Windows 10.0.19045
* Python 3.9.1
* zipfile (built-in), pandas 1.2.2, numpy 1.22.2, sklearn 1.0.1, keras 2.12.0

## Execution - time series analysis example
```python
from timeseriesanalysis import *

Data.extract_data('C:\\Users\\User\\Desktop\\hungarian+chickenpox+cases.zip', 'hungary_chickenpox.csv', 'C:\\Users\\User\\Desktop')          
chickenpox_data = Data()
chickenpox_data.read_data('C:\\Users\\User\\Desktop\\hungary_chickenpox.csv')
chickenpox_data.clean_data('Date', 'BUDAPEST', 0, 1000) 

chickenpox_nn = NeuralNetwork()
chickenpox_nn.split_data(chickenpox_data.df_clean, "27/08/2012")  
chickenpox_nn.define_model()                                  
chickenpox_mae_rmse = chickenpox_nn.fit_model()
print(chickenpox_mae_rmse)
```


 
