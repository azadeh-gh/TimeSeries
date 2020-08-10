# Using deep learning neural networks to Predict the recovery effectivness (REN) of water well
**This repository contains project data, notes and scripts for  artificial neural network based predicting of REN using factors that significantly affecting REN in water well.**  
## Description
The REN performance criterion equals the amount of the water injected in water well that is recoverable via the same well during a specified exatraction period.

The dataset used to train and verify the ANN derive from 5,000 MODFLOW-MT3DMS simulations. Inputs (independent variables) for these simulations differ for seven hydrogeological and operational
factors that potentially impact REN. These factors are: background gradient, hydraulic conductivity, injection rate and duration, extraction rate and duration, storage duration, aquifer thickness, porosity, and
longitudinal dispersivity. In MODFLOW-MT3DMS simulations extraction begins after storing the injected water in well for 12 months.
## Source
Data has been recieved from the author.

https://www.sciencedirect.com/science/article/pii/S0022169418304645?via%3Dihub

data folder consist of the data were used in this project
## Installation
* Python 2.7
* tensorflow 2.0
* sklearn
* statsmodels
* pandas
* matplotlib
## Solution
Herein, we describe the use of Python packages for this project.

**Packages import:**
```python
import numpy as np
import tensorflow as tf
import os
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import pickle
```
**Preprocessing of data**

reading raw data
```python
raw_csv_data = pd.read_csv("data/Data.out",delim_whitespace=True)
```
Create a "ratio" column by diviving the values of "Ext" and "Inj" columns
```python
raw_csv_data["ratio"]=raw_csv_data["Ext"]/raw_csv_data["Inj"]
```
Standrizing data using sklearn preprocessing 
```python
scaled_data = preprocessing.scale(raw_csv_data)
```
The results of preprocessing are as list type, We converted them to DataFrame with previous column names for them
```python
Scaled_DF=pd.DataFrame(scaled_data,columns=raw_csv_data.columns)
```
Now we can define dependent(y) and independents(x) columns
```python
y=Scaled_DF['REN_3_2']
x1=Scaled_DF[['K', 'Inj', 'Por', 'b', 'CHD', 'ratio', 'DSP']]
```
We can use statsmodel to see the OLS Regression Results summary 
```python
x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary())
```
Using p values from the results of summary we relized "DSP" does not have significant effect on y values, so we do not need this column values for prediction.
We drop the "DSP" column
```python
Scaled_DF=raw_csv_data.drop(["DSP"],axis=1)
```
Define the Inputs and Target columns for our model
```python
scaled_inputs=Scaled_DF.iloc[:,0:6]
targets_all=Scaled_DF.iloc[:,6:]
```
Changing the type of target_all to mumpy array
```python
targets_all=targets_all.values
```
Shuffle the data
```python
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)
```
Use the shuffled indices to shuffle the inputs and targets.
```python
shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_all[shuffled_indices]
```
Count the total number of samples
```python
samples_count = shuffled_inputs.shape[0]
```
Split the dataset into train, validation, and test.

Count the samples in each subset, assuming we want 80-10-10 distribution of training, validation, and test.
Naturally, the numbers are integers
```python
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)
```
The 'test' dataset contains all remaining data.
```python
test_samples_count = samples_count - train_samples_count - validation_samples_count
```
Create variables that record the inputs and targets for training.

In our shuffled dataset, they are the first "train_samples_count" observations.
```python
train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]
```
Create variables that record the inputs and targets for validation.

They are the next "validation_samples_count" observations, folllowing the "train_samples_count" we already assigned.
```python
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]
```
Create variables that record the inputs and targets for test.
They are everything that is remaining.
```python
test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]
```
Save train, validation and test data as npz file
```python
np.savez(os.path.join(Dir,'data_train'), inputs=train_inputs, targets=train_targets)
np.savez(os.path.join(Dir,'data_validation'), inputs=validation_inputs, targets=validation_targets)
np.savez(os.path.join(Dir,'data_test'), inputs=test_inputs, targets=test_targets)
```
**Building blocks of machine learning algotithm**

let's create a temporary variable npz, where we will store each of the three Audiobooks datasets
```python
npz = np.load(os.path.join(Dir,'data_train.npz'))
```
to ensure that they are all floats, let's also take care of that
```python
train_inputs = npz['inputs'].astype(np.float)
train_targets = npz['targets'].astype(np.float)
```
we load the validation data in the temporary variable
```python
npz = np.load(os.path.join(Dir,'data_validation.npz'))
```
we can load the inputs and the targets in the same line
```python
validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.float)
```
 we load the test data in the temporary variable
 ```python
 npz = np.load(os.path.join(Dir,'data_test.npz'))
```
we create 2 variables that will contain the test inputs and the test targets
```python
test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.float)
```
**Model**

Set the input and output sizes
```python
input_size = 6
output_size = 8
```
Use same hidden layer size for all hidden layers. Not a necessity.
```python
hidden_layer_size =200
```
define how the model will look like
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'), 
    tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'),
    tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'),
    tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'),
    tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'),
    tf.keras.layers.Dense(output_size, activation='linear')])
```
**Choose the optimizer and the loss function**
```python
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
```
**Training**
set the batch size
```python
batch_size = 100
```
set a maximum number of training epochs
```python
max_epochs = 100
```
set an early stopping mechanism.
```python
early_stopping = tf.keras.callbacks.EarlyStopping(patience=1)
```
**Fit the model**
```python
model.fit(train_inputs, 
          train_targets, 
          batch_size=batch_size, 
          epochs=max_epochs,
          callbacks=[early_stopping], 
          validation_data=(validation_inputs, validation_targets),
          verbose = 2 )
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
print('\nTest loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))
```
**Test result**

Test loss: 0.02. Test accuracy: 98.20%

The Test result of Test accuarcy are good.

**plot**

Plot the model prediction vs measured data for one of the outputs test data.
```python
predictions = model.predict(test_inputs)
print("predictions shape:", predictions.shape)

plt.scatter(test_targets[:,6]*100,predictions[:,6]*100,color="blue")
plt.plot([0,100],[0,100],color="red")
plt.xlabel("Measured")
plt.ylabel("Predicted")
plt.show()
```
![](Results/image.png?raw=true)

now that the residual plot looks good we can save the model for future uses.

**Save the model**
```python
model.save("model.h5")
```
