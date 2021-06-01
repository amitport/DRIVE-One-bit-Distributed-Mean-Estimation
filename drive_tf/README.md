# DRIVE: TensorFlow implementation

### install requirements:

```setup
pip install -r requirements.txt
```

### usage 

```python
from drive_tf import drive_encoder, drive_plus_encoder
```
This package exports DRIVE and DRIVE plus encoders designed to work with the [TensorFlow Federated](https://www.tensorflow.org/federated/) framework. See the following [tutorial](https://github.com/tensorflow/federated/blob/b059f263bfaa22aadf249d2a801ed9dcd6c68bac/docs/tutorials/tff_for_federated_learning_research_compression.ipynb) for a demonstration of encoder usage in TensorFlow Federated.

See `experiments/federated` for additional usage examples.