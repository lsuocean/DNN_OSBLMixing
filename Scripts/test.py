import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
import numpy as np
import xarray as xr
import os
import statistics as st
from os import listdir
import scipy.io as sio
from datetime import datetime

ctn_pth='../input/test/'
fname=listdir(ctn_pth)


# --------------------- DNN super-parameters ------------------------
nz = 32      # number of vertical grids
n_depth = 2  # number of hidden layers
units = 128   # number of units in each hidden layer
steps = 2
# How many time steps of forcing condition inclued.
#   1: forcing at current time point only
#   2: forcing at both current and one previous time points
#   3: forcing at both current and two previous time points

n_case = 38 # number of testing periods
n_size = 10 # ensemble case size
time_step = 1800 # time step (second)

# ------------- input | output layer setup --------------------------------
input_dim = nz*(2+steps*2)+4*steps
# input array includes:
#   1. T & S profiles at current time point [nz * 2]
#   2. Stokes profiles in x & y directions  [nz * 2 * steps]
#   3. Surface heat flux & Wind stress vector [3 * steps]
#   Note: [input_dim = nz*2 + 6*steps] if using surface stokes information only
#         [input_dim = nz*2 + 4*steps] if no stokes information is included into inputs

output_dim = nz*2
# output array includes:
#     predicted dT & dS profiles [nz * 2]

# -------------- model building --------------------------------------
def model_ocean(input_dim,n_depth,output_dim,units):
    model = keras.Sequential()
    model.add(layers.Dense(units,input_dim=input_dim,
        kernel_initializer=initializers.RandomNormal(stddev=0.05),
        bias_initializer=initializers.Zeros()))
    model.add(layers.LeakyReLU(alpha=0.1))
    for i in range (0,n_depth):
        model.add(layers.Dense(units,
            kernel_initializer=initializers.RandomNormal(stddev=0.05),
            bias_initializer=initializers.Zeros()))
        model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dense(output_dim,
        kernel_initializer=initializers.RandomNormal(stddev=0.05),
        bias_initializer=initializers.Zeros()))
    return model

# --------------- load normalization parameters -----------------------
#
#    The predicted dT & dS by DNN are normalized. The parameters are used to retrive 
# the true magnitude of dT & dS

paras = np.load('../../input/norm_paras.npz')
m_paras = paras['dt_mean']
sd_paras = paras['dt_std']

# -------------- get a list of files that contain the truths related to the testing periods

# -------------- DNN prediction loops ---------------------------------
ii = range(0,n_case)

    ## obtain the input data and the truth for period ii ##
    i = fname[ii]
    data_test  = xr.open_dataset(ctn_pth+i)
    data_flow = np.array(data_test['data'])[:,:input_dim]
    test_truth = np.array(data_test['data'])[:,input_dim:]
    test_les = data_flow - data_flow + data_flow
    test_dnn = data_flow - data_flow  # variable to store updated T & S based on predicted dT & dS by DNN
    mt=model_ocean(input_dim,n_depth,output_dim,units)

    ## the predicted dT & dS by DNN at the time step [j-1] shall be used to update the T & S profile at the time step [j]. 
    ## The updated T & S at the time step [j] will be used to predict dT & dS at the time step [j]
    for j in range(data_flow.shape[0]):
        if j>0:
            dnn_pre_tmp = test_dnn[(j-1),0:(nz*2)]*sd_paras+m_paras
            data_flow[j,:(nz*2)] = data_flow[(j-1),:(nz*2)]+dnn_pre_tmp*time_step
        test_dnn0 = np.zeros((n_size,output_dim))


### Let DNN make 10 different predictions based on  different best-weight [h5] files loaded
        for k in range(n_size):
            pth_root='../weights/weights_best_1.h5'
            mt.load_weights(pth_root)
            test_dnn0[k,:] =  mt.predict(data_flow[j,:].reshape(1,-1))

### Get the median value of predictions by all the DNNs at each time step
        for m in range(output_dim):
            test_dnn[j,m] = st.median(test_dnn0[:,m])

    ## Save the outputs
    adict ={}
    adict['les'] = test_les
    adict['dnn'] = data_flow
    out_pth = '../output/'
    if not os.path.exists(out_pth):
    os.mkdir(out_pth)
    root_path = out_pth+'/test_'
    formatt = '.mat'
    save_fname=i[5:-3]
    sio.savemat(root_path+save_fname+formatt,adict)
