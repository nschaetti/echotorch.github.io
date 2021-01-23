
# Gettings started

## Reservoir Computing and time series analysis made simple

EchoTorch is a Python package which allow you to easily implement Reservoir Computing
models such as Echo State Networks, more complex ones such as Conceptors and 
DeepESN but also all neural network models working on timeseries. EchoTorch is a 
PyTorch for Reservoir Computing and timeseries analysis.

## Overview

## Installation and import

EchoTorch is available on pip and pip test. To install it, just call pip

> pip install EchoTorch

or if you want to test the development version

> pip install -i https://test.pypi.org/simple/ EchoTorch

## Getting started


### Datasets

```python
narma10_training_dataset = echotorch.dataset("narma-10")
narma10_test_dataset = echotorch.dataset("narma-10")
```

### Preprocessing

```python
normalize_transformer = echotorch.transformer("normalize")
narma10_training_dataset.transform = normalize_transformer
narma10_test_dataset.transform = normalize_transformer
```

### Create basic Echo State Network

```python
esn_model = etnn.LiESN(...)
```

### Train the network

> echotorch.fit(esn_model, narma10_training_dataset)

### Eval the model

> nrmse_score = echotorch.eval(esn_model, narma10_test_dataset)
> print(nrmse_score)

## Summary

```python
import echotorch
import echotorch.nn as etnn

# Create training and test sets
narma10_training_dataset = echotorch.dataset("narma-10")
narma10_test_dataset = echotorch.dataset("narma-10")

# Create transformer to normalize time series
normalize_transformer = echotorch.transformer("normalize")
narma10_training_dataset.transform = normalize_transformer
narma10_test_dataset.transform = normalize_transformer

# Create Leaky integrator ESN
esn_model = etnn.LiESN(...)

# Train the ESN
echotorch.fit(esn_model, narma10_training_dataset)

# Eval trained ESN and show the NRMSE
nrmse_score = echotorch.eval(esn_model, narma10_test_dataset)
print(nrmse_score)
```

## Next


