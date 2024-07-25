![alt text](image.png)
# PIINs-Harmonic-Oscillator

![Project Image](image.png)

## Overview
This repository contains the implementation of a Physics-Informed Neural Network (PINN) to solve the 1D damped harmonic oscillator problem. The PINN approach leverages the underlying physical laws described by differential equations to enhance the learning process, especially when data is scarce or noisy.

## Problem Statement
The example problem solved here is the 1D damped harmonic oscillator:
$$
m \dfrac{d^2 x}{d t^2} + \mu \dfrac{d x}{d t} + kx = 0
$$
with the initial conditions
$$
x(0) = 1, \quad \dfrac{d x}{d t} = 0
$$

We focus on solving the problem for the under-damped state, i.e., when 
$$
\delta < \omega_0, \quad \text{where} \quad \delta = \dfrac{\mu}{2m}, \quad \omega_0 = \sqrt{\dfrac{k}{m}}
$$
The exact solution is:
$$
x(t) = e^{-\delta t}(2 A \cos(\phi + \omega t)), \quad \text{with} \quad \omega=\sqrt{\omega_0^2 - \delta^2}
$$


## Project Structure
- **Harmonic oscillator PINN.ipynb**: Jupyter Notebook containing the complete code and detailed explanation.
- **main.py**: Main script for training the model.
- **model.py**: Definition of the Fully Connected Neural Network (FCN) class.
- **config.py**: Configuration settings for the hyperparameters.
- **requirements.txt**: List of dependencies required for the project.
- **image.png**: Diagram or visual representation related to the project.
- **README.md**: Project documentation (this file).

## Getting Started

### Prerequisites
- Python 3
- [Anaconda](https://www.anaconda.com/products/distribution) (recommended)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/indigo14/PIINs-Harmonic-Oscillator.git
    cd PIINs-Harmonic-Oscillator
    ```

2. Create and activate a conda environment:
    ```bash
    conda create -n pinn python=3
    conda activate pinn
    ```

3. Install the required dependencies:
    ```bash
    conda install jupyter numpy matplotlib
    conda install pytorch torchvision torchaudio -c pytorch
    ```

4. Alternatively, you can install dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### Jupyter Notebook
Open the `Harmonic oscillator PINN.ipynb` notebook to explore the code and reproduce the results:
```bash
jupyter notebook Harmonic oscillator PINN.ipynb
