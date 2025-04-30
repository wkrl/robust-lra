# Robust-LRA

This repository contains a NumPy implementation of the RobustLRA algorithm proposed in the paper:

> Arora, Raman, and Jalaj Upadhyay. "Differentially private robust low-rank approximation." Advances in neural information processing systems 31 (2018).

This implementation allows for the computation of a low-rank matrix while ensuring differential privacy. It uses Gaussian distributions for the generation of the sketch matrices $\Phi, \Psi, S$ and $T$, as well as a Laplace distributions for the noise matrices $N_1$, $N_2$ and $N_3$. 

### Setup

1.  Ensure you have Python 3 installed on your system.
2.  Install the necessary dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

```python
from RobustLRA import SketchShapes, RobustLRA
import numpy as np


robust_lra = RobustLRA(
    A=np.array([...]),
    sketch_shapes=SketchShapes(...),
    eps=...,
    k=...,
) 
M = robust_lra.compute()
```