# tszpower: Cosmological Parameter Inference with tSZ Power Spectrum

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![JAX](https://img.shields.io/badge/JAX-%23F37623.svg?logo=jax&logoColor=white)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


A repository for computing the thermal Sunyaev-Zel'dovich (tSZ) power spectrum and performing cosmological parameter inference using both **Markov Chain Monte Carlo (MCMC)** and **Simulation-Based Inference (SBI)**. Built with JAX for accelerated computation and automatic differentiation.

## Key Features
- **tSZ Power Spectrum Computation**: Fast calculation of the tSZ power spectrum using JAX for GPU acceleration.
- **Cosmological Parameter Inference**:
  - Traditional MCMC sampling with `cobaya`
  - Modern simulation-based inference (SBI) with neural density estimators
- **JAX-powered**: Automatic differentiation and hardware acceleration (CPU/GPU/TPU) support.
- **Modular Design**: Easily extensible for testing different cosmological models or inference methods.

<!-- 
## Installation
```bash
git clone https://github.com/licongxu/tszpower.git
cd tszpower
pip install -r requirements.txt
 -->
## Installation

To install tszpower in editable mode, clone the repository and run:

```bash
pip install -e .
```

## Usage

Example usage of the code is provided in the [examples](examples) folder. In addition, the code is built upon the notebooks found in the [notebook](notebook) folder.

