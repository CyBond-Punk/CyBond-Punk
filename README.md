# CyBond-Punk
🌟 Welcome to the repository for our paper titled "**CyBond Punk: Rethinking Message Passing Mechanism via Graph Edge Space**".

Our implementation is based on [PyTorch Geometric](https://www.pyg.org/) that are provided under MIT License. All experiments were run in a shared computing cluster environment with varying CPU and GPU architectures. These involved a A100 (40GB) GPUs. The resource budget for each experiment was 1 GPU, between 4 and 6 CPUs, and up to 32GB system RAM.

## 🗂️ CyBond-Punk Directory
```shell
- CyBond-Punk
  - eval_edge.py         # Evaluation script for models with edge information
  - model_edge.py        # Model definition with edge information
  - model_noedge.py      # Model definition without edge information
  - requirements.txt     # List of dependencies
  - train_edge.py        # Training script for models with edge information
  - train_noedge.py      # Training script for models without edge information
  - utils.py             # Utility functions
  - README.md            # This readme file

```

## 🛠️ Installation

To get started with CyBond-Punk, follow these installation instructions:

### Prerequisites

Ensure you have the following software installed:

- Python 3.8 or higher
- pip (Python package installer)

### Clone the Repository

Clone this repository to your local machine using:

```sh
git clone https://github.com/yourusername/CyBond-Punk.git
cd CyBond-Punk
```

### Install Dependencies
Install the required Python packages:
```shell
pip install -r requirements.txt
```

Since our implementation relies on PyTorch Geometric, you'll need to install it separately. Use the following command to install PyTorch Geometric along with the CUDA support suitable for your system:

```shell
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
```

## ▶️ Usage
After installing the necessary dependencies, you can run the various scripts provided in this repository:

### Training Models
To train the model with edge information: (such as **ZINC** and **LRGB** datasets)
```shell
python train_edge.py
```
To train the model without edge information: (such as **TUDataset** datasets)
```shell
python train_noedge.py
```

### Evaluating Models
To evaluate the model with edge information:
```shell
python eval_edge.py
```

## License
This project is licensed under the **Apache License 2.0** - see the LICENSE file for details.

## Acknowledgements
This work was made possible thanks to the resources provided by the shared computing cluster environment.

## Contributing
We welcome contributions! Please read our contributing guidelines for more details.

## Contact
If you have any questions or issues, please open an issue in this repository or contact us at amihua@mail2.gdut.edu.cn.

Happy coding! 🚀