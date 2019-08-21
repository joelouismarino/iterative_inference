# Iterative Amortized Inference

Code to accompany the paper [Iterative Amortized Inference](https://joelouismarino.github.io/files/papers/2018/iterative_amortized_inference/icml_2018_paper.pdf) by Marino et al., ICML 2018.


## Installation & Set-Up

First, clone the repository by opening a terminal and running:
```
$ git clone https://github.com/joelouismarino/iterative_inference.git
```
The code uses PyTorch version `0.3.0.post4` and visdom version `0.1.7`. To avoid conflicts with more recent versions of these packages, you may wish to create a conda environment:
```
$ conda create --name it_inf python=2.7
```
To enter the environment, run:
```
$ source activate it_inf
```
Within the environment, install PyTorch by visiting the list of versions [here](https://pytorch.org/previous-versions/), and grabbing version `0.3.0.post4` for your version of CUDA (`8.0`, `9.0`, `9.1`, etc.). Note that the code requires CUDA. Be sure to also install torchvision.

To install vidsom, run
```
(it_inf) $ pip install visdom==0.1.7
```
You will also need to install dill, a serialization package, and scipy:
```
(it_inf) $ pip install dill scipy
```
To exit the environment, run:
```
(it_inf) $ source deactivate
```

## Running the Code

To use visdom for plotting, open an terminal and run
```
python -m visdom.server
```
Note that if you created a conda environment to use visdom version `0.1.7`, you will need to enter that environment before activating visdom (see above).

The code can be run from the terminal using command line arguments. The arguments are
* `dataset`,
* `model_type`,
* `inference_type`,
* `data_path`, and
* `log_path`.

For instance, to run a single-level model on MNIST with iterative inference, run:
```
python main.py --dataset 'mnist' --model_type 'single_level' --inference_type 'iterative' --data_path '/path/to/data/' --log_path '/path/to/logs'
```
Be sure to replace the paths for `data_path` and `log_path` with valid paths to where the data and logs should be saved, respectively.

You can watch the training progress by opening a browser window and navigating to `http://localhost:8097`, and selecting the visdom environment corresponding to the experiment.
