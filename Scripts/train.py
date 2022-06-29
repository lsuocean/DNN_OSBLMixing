import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
import numpy as np
import xarray as xr
import os
import time

# -------------- DNN super-parameters -------------------------------
nz = 32      # number of vertical grids
n_depth = 2  # number of hidden layers
units = 128  # number of units in each hidden layer
epoch = 2000 # epochs of DNN model

steps = 2
# How many time steps of forcing condition inclued.
#        1: forcing at current time point only
#        2: forcing at both current and one previous time points
#        3: forcing at current time point and two previous time points

# ------------- input | output layer setup --------------------------------
input_dim = nz*(2+steps*2)+4*steps
# input array includes:
#    (1). T & S profiles at current time point [nz * 2]
#    (2). Stokes profiles in x & y directions  [nz * 2 * steps]
#    (3). Surface heat flux & Wind stress vector & z [3 * steps]
#  Note: [input_dim = nz*2 + 6*steps] if using surface stokes information only
#        [input_dim = nz*2 + 4*steps] if no stokes information is included into inputs

output_dim = nz*2
# output array includes:
#     predicted dT & dS profiles [nz * 2]


# -------------- Load data -------------------------------------------
data_train = xr.open_dataset('../input/ocean_train.nc')
data_valid = xr.open_dataset('../input/ocean_valid.nc')
train_flow = np.array(data_train['data'])[:,:input_dim]
train_truth = np.array(data_train['data'])[:,input_dim:]
valid_flow = np.array(data_valid['data'])[:,:input_dim]
valid_truth = np.array(data_valid['data'])[:,input_dim:]

# -------------- model building --------------------------------------
model = keras.Sequential()
model.add(layers.Dense(units,input_dim=input_dim,kernel_initializer=initializers.RandomNormal(stddev=0.05),
bias_initializer=initializers.Zeros()))
model.add(layers.LeakyReLU(alpha=0.1))
for i in range (0,n_depth):
    model.add(layers.Dense(units,kernel_initializer=initializers.RandomNormal(stddev=0.05),bias_initializer=initializers.Zeros()))
    model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.Dense(output_dim,kernel_initializer=initializers.RandomNormal(stddev=0.05),bias_initializer=initializers.Zeros()))

## loss function
def loss_wan(y_true, y_pred):
    delta_y=tf.reduce_sum((y_true-y_pred)**2,[1],keepdims=True)
    loss0 = tf.reduce_mean(delta_y)

    # additional panelty for violating the conservation law
    paras = np.load('../input/norm_paras.npz')
    z0 = paras['z']
    m_paras0 = paras['dt_mean']
    sd_paras0 = paras['dt_std']
    z = z0[[np.arange(3,128,4)]]
    z = np.append(0,z)
    m_paras = m_paras0[0,260:324]
    sd_paras = sd_paras0[0,260:324]
    #z = np.append(0,z1)
    #z = np.array(data_z['grids_z']).reshape(-1,1)
    dz = z[1:]-z[:-1]
    sum_dz = np.sum(dz)

    dy_predict = tf.math.subtract(y_pred,y_true)
    dy_tmp = tf.math.multiply(dy_predict,sd_paras)
    dy_true = tf.add(dy_tmp,m_paras)

    dy_t_add = tf.math.multiply(dy_true[:,0:32],dz)
    dy_s_add = tf.math.multiply(dy_true[:,32:64],dz)
    dy_t_sum = tf.reduce_sum(dy_t_add,[1],keepdims=True)
    dy_s_sum = tf.reduce_sum(dy_s_add,[1],keepdims=True)
    loss1 = tf.math.divide(dy_t_sum,sum_dz)
    loss2 = tf.math.divide(dy_s_sum,sum_dz)
    loss_conserv = tf.reduce_sum(tf.add(loss1,loss2))

    # 0.1 is the panelty weight parameter and can be modified before the training process
    loss = tf.add(loss0,loss_conserv*0.1)
    return loss

# -------------- model configuration ------------------------------
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.6e-3,decay_steps=2000,decay_rate=0.9)
model.compile(loss=loss_wan,optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))

# ------------- model training -------------------------------------
if not os.path.exists('../weights/'):
    os.mkdir('../weights/')
wt_path='../weights/'
if not os.path.exists(wt_path):
    os.mkdir(wt_path)
wt_fname = wt_path + 'weights_best.h5'
# name of the file to record the best weight during training process 

checkpoint=keras.callbacks.ModelCheckpoint(wt_fname,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
# only record the best weight when the loss function is the minimum 

callback_list =[checkpoint]
history=model.fit(train_flow,train_truth,epochs=epoch,batch_size=2048,callbacks=callback_list,verbose=2,validation_data=(valid_flow,valid_truth))

# ----------- save training and validation losses over epochs -----------
history_dict=history.history
loss_values = np.array(history_dict['loss']).reshape(-1,1)
val_loss_values = np.array(history_dict['val_loss']).reshape(-1,1)
epochs = np.array(range(1, len(loss_values) + 1)).reshape(-1,1)
if not os.path.exists('../epoch/'):
   os.mkdir('../epoch/')
ep_path = '../epoch/'
if not os.path.exists(ep_path):
   os.mkdir(ep_path)
ep_fname = ep_path + 'cong_vs_epoch.dat'
np.savetxt(ep_fname,np.concatenate((epochs, loss_values, val_loss_values), axis=1))

