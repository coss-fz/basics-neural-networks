# Neural Networks: Basic Concepts
This repository is for a project that involves the construccion of a basic neural network (MLP), and the performance comparison against the commonly used libraries (Py Torch and Tensorflow Keras).
If you aren't familiar with neural networks, go to ..., where I explain the basics concepts with a MLP.


## Quick Start
In order to run all the codes succesfully, you'll need to create a python virtual environment and install all the necessary requirements.

### Configure the Python Environment
1. Create the virtual environment (there is no mandatory version, but `python 3.9` is recommended)
   ```sh
   python -m venv <venv_name>
   ```
2. Activate the environment
   ```sh
   source <venv_name>/bin/activate
   ```
   Whenever you want to get out from the environment, just run the next command:
   ```sh
   deactivate
   ```

### Install the required python libraries and packages
1. Install the necessary libraries through the 'Makefile'
   ```sh
   pip install -r requirements.txt
   ```


## Navigate through the project
Once you are all set, feel free to browse through the project. 

* Go to 'src' to see the MLP class and its functions
* Go to 'test' to see the functions used for the unit tests, and try to test the code yourself
   ```sh
   pytest tests/test_neural_network.py
   ```
* Go to the Jupyter Notebooks which contain the examples, and run them yourself
