## Description 
Using a neural network for time series analysis. The project involves processing the data so that the format is conducive for downstream analysis. Then go on and build a neural network and train the model, and then test out new images to see if the trained model works??. 

The `DataPrep` class has a `read_zip()` method for reading in a zip file,  and a `clean_dataframe()` method for cleaning and filtering allowed values. The `NeuralNetwork` class has a `test_train_split()` method for using a time stemp to split the data, a `model_structure` method for defining the structure recursive artifical neural model,  and a `model_performance` method for training the model and evaluating its performance.   

## Dependencies
* Microsoft Windows version 10.0.19045
* Python version 3.9.1
* Pandas, Numpy, Matplotlib, Sklearn, Keras, Zipfile, Urllib, Shutil, Os 

## Execution - Infectious Disease Example
```python
s = "Python syntax highlighting"
print s
```

## Animation - Infectious Disease Example
remember to add the link to the GIF, which I must also make sure to add to the repo, see stackoverflow - not sure if I add the prediction
