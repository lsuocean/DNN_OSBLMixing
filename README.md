# DNN_OSBLMixing

This repository is the computer codes to train and test a deep neural network (DNN) model for vertical mixing by ocean surface boundary layer turbulence described in Liang et al. [2022]. 

## A.  Download deep learning codes

Create a local copy of the codes by 

```
git clone https://github.com/lsuocean/DNN_OSBLMixing/
```

There are two files under the directory 

- **train.py** – codes to train a DNN model

- **test.py**   – codes to run an offline 1D prediction for 9 days, using the trained DNN model.

The codes are written in Python and use Python packages "tensorflow" and "keras"

## B.  Create training and testing data

Create training/testing datasets using either observational data or numerical solutions. In Liang et al. [2022], we use multi-year large-eddy simulation (LES) solutions for the Ocean State Papa as training data (see Liang et al. [2022] for details). If you are interested in our solutions, feel free to send me an email to Dr. Jun-Hong Liang ([jliang@lsu.edu](mailto:jliang@lsu.edu)). 

The datasets include
- **ocean_train.nc**: the training dataset
- **ocean_valid.nc**: the validation dataset
- **norm_paraz.npz**: our DNN model predicts normalized temperature and salinity change rate over time. This file contains parameters to convert normalized outputs to the true values.

## C.  Train the DNN model

- Run the training codes

```
python3 codes/train.py
```

After the training is completed, two files are created:
1. `cong_vs_epoch.dat`: a record of the training and validation losses over epoch during DNN training process
2. `weights_best.h5`: the best trainable DNN parameters during the training process.


## D.  Test the DNN model
- Run the prediction codes

```
python3 codes/test.py
```

Steps C and D are repeated to select the optimal model hyperparameters.

------

Details of the model and its testing is in:
Liang, J.H., Yuan, J., Wan, X., Liu, J., Liu, B., Jang, H. and Tyagi, M., 2022. Exploring the use of machine learning to parameterize vertical mixing in the ocean surface boundary layer. *Ocean Modelling*, p.102059. [doi.org/10.1016/j.ocemod.2022.102059](https://doi.org/10.1016/j.ocemod.2022.102059)

If you have any questions regarding the model, please feel free to open an issue or send an email to Dr. Jun-Hong Liang ([jliang@lsu.edu](mailto:jliang@lsu.edu)).
