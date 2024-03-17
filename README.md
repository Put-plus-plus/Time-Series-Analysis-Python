## Description 

In time series analysis a time series, ie a series of data points indexed in time order, is analysed in order to extract meaningful characteristics of the data. In this case the time series analysis is organised into two classes, `Data` that holds the pre-processing of the data and `NeuralNetwork` that holds the recursive neural network model of the data. The `Data` class has three methods: the `extract_data()` method that extracts an uncompressed csv file from a zip file, the `read_data()` method that retrieves raw data from a csv file in the form of a pandas data frame, and the `clean_data()` method that removes anomalies from the data and outputs a dataframe with a single column of counts of events and date data as the dataframe index. The `NeuralNetwork`class has three methods: the `split_data()` method that splits the data into four numpy arrays with train and test data, the `define_model()` method that defines the structure of the recursive neural network model, and the `fit_model()` method that fits the recursive neural network model to the data and returns the mean absolute error (mae) and root mean squarred error (rmse).

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


 
