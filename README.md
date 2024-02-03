## Description 
Using a neural network for time series analysis. The project involves processing the data so that the format is conducive for downstream analysis. Then go on and build a neural network and train the model, and then test out new images to see if the trained model works??. 

The `DataCleaning` class has a `read_zip()` method for reading in a zip file,  and a `clean_dataframe()` method for cleaning and filtering allowed values. The `NeuralNetwork` class has a `model_data_split()` method for using a time stemp to split the data, a `model_structure()` method for defining the structure recursive artifical neural model,  and a `model_performance()` method for training the model and evaluating its performance.   

 
## Dependencies
* Microsoft Windows 10.0.19045
* Python 3.9.1
* zipfile (built-in), pandas 1.2.2, numpy 1.22.2, sklearn 1.0.1, keras 2.12.0

## Execution - chickenpox incidence example
```python
chicken_pox_model = NeuralNetwork()
print chicken_pox_model.model_performance()
# OUTPUT: something
```

## Animation - chickenpox incidence example
remember to add the link to the GIF, which I must also make sure to add to the repo, see stackoverflow - not sure if I add the prediction

 
