<div align = "center"><h1>welcome to ai_utils</h1></div>
<div align="center">
# ai_utils (ai_lib)
**High-Performance, Cython-Optimized Machine Learning Utilities**
PyPI version

Documentation

Python Version

License
Portfolio • Documentation • Report Bug
</div>
## Overview
ai_utils (available on PyPI as ai_lib) is a lightning-fast machine learning utility library. Built with **Cython (.pyx)** under the hood, it provides highly optimized implementations of core mathematical operations necessary for deep learning, neural networks, and transformer architectures.
Say goodbye to slow Python loops and harness the power of C-level execution speeds with the simplicity of Python!
## Key Features
 * **Cython-Powered:** Core modules are written in .pyx for maximum performance.
 * **Transformer Ready:** Built-in fast Positional Encoding (pos_enc).
 * **Optimized Math:** Standard and ultra-fast matrix dot products.
 * **Lightweight:** Minimal dependencies, easy to integrate into existing AI pipelines.
## Installation
Install the library directly from PyPI using pip:
```bash
pip install ai_lib

```
## Quick Start
You can easily import everything from the library to get started immediately:
```python
from ai_lib import *
import numpy as np

```
### Module Highlights
Here is a quick look at what you can do with the included Cython modules:
#### 1. Matrix Operations (dot & dot_fast)
Perform standard or highly-optimized matrix multiplications.
```python
a = np.random.randn(1000, 1000)
b = np.random.randn(1000, 1000)

# Ultra-fast dot product
result = dot_fast(a, b) 

```
#### 2. Activation Functions (activation)
Apply fast non-linearities to your tensors.
```python
# Assuming 'result' is your matrix from the previous step
activated_output = activation(result, type='relu') 

```
#### 3. Softmax Probabilities (softmax)
Efficiently compute softmax along the desired axis.
```python
probabilities = softmax(activated_output)

```
#### 4. Positional Encoding for Transformers (pos_enc)
Generate sine and cosine positional encodings for sequence models in a flash.
```python
seq_length = 512
embed_dim = 64

# Generate encodings instantly
encodings = pos_enc(seq_length, embed_dim)

```
## Documentation
For deep dives, API references, and advanced usage examples, please check out the official documentation:
**ai_lib Official Documentation**
## Author & Support
This project is actively developed and maintained by **Ansh Raj**.
 * **Website / Portfolio:** Ansh Studios
 * **Email:** anshraj0000000001@gmail.com
 * **GitHub:** @ANSH-ind
If you find this library helpful, please consider giving it a star on GitHub!
<div align="center">
<i>Built for the AI community by Ansh Raj.</i>
</div>
