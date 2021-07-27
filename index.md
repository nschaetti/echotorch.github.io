
EchoTorch is a Python package intended to simplify the evaluation and implementation of 
machine learning models based on Echo State Networks and Reservoir Computing.

It allows to implement more advanced models based on Conceptors and on the Deep Learning 
paradigm such as DeepESN. You will also be able to integrate random recurring models 
into your pytorch models and simply generate time series data for your research.

---

## The TimeTensor class

The basic element of EchoTorch is the TimeTensor, it is an extension of the Tensor class of pytorch but 
specifying a special dimension as being temporal in nature. You could specify a batch and a channel 
dimension as well. A TimeTensor as a size of `(batch size, n. channels, time length, ...)`, if the TimeTensor 
has no batch dimension and no channels, the first dimension should be the time dimension. Dimensions after 
the time dimension are seen as the size of the timeseries.

Let's create a TimeTensor full of zeros, similarly as torch:

`x = echotorch.zeros((10,), 100)`

The variable `x` is a 10-dimensional timeseries of length 100 without batch and channel dimension.
