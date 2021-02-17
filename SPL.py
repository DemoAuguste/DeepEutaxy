import tensorflow as tf
import numpy as np
import argparse
import os
from keras import backend as K
from models import *
from utils import *
from settings import *
from CLR import CyclicLRCustom

tf.config.experimental_run_functions_eagerly(True) 
v = None
threshold = -0.1
growing_factor = 1.3
ce_loss = tf.keras.losses.CategoricalCrossentropy()
current_batch_index = None

def spl_aug(super_loss):
    global threshold
    v = super_loss > threshold
    return v.numpy().astype(np.int)

@tf.function
def spl_loss(y_true, y_pred):
    global current_batch_index, v

    h_idx = np.argmax(y_true.numpy(), axis=1).flatten().reshape(-1,1)
    line = np.arange(y_true.numpy().shape[0]).reshape(-1,1)
    index = np.hstack((line, h_idx))

    super_loss = - y_pred * y_true
    super_loss = tf.gather_nd(super_loss, index)
    temp_v = spl_aug(super_loss)
    v[current_batch_index] = temp_v.flatten()
    spl = tf.reduce_mean(super_loss * temp_v)

    return spl

    
def set_global_v(n_samples):
    global v
    v = np.zeros(n_samples, dtype=np.int) 

def set_current_batch_idx(value):
    global current_batch_index
    current_batch_index = value

def increase_threshold():
    global threshold, growing_factor
    threshold *= growing_factor



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='resnet20')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10')
    parser.add_argument('-v', '--version', type=str, default='1')
    parser.add_argument('-c', '--clr', type=int, default=0)

    args = parser.parse_args()

    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    ver = args.version
    useCLR = args.clr

    model_weights_dir = os.path.join(weights_dir, model_name)
    model_weights_dir = os.path.join(model_weights_dir, dataset)
    model_weights_dir = os.path.join(model_weights_dir, ver)

    if not os.path.exists(model_weights_dir):
        os.makedirs(model_weights_dir)

    trained_weights_path = os.path.join(model_weights_dir, 'spl_trained.h5')

    model, optimizer = get_model(model_name, False)
    x_train, y_train, x_test, y_test = get_dataset(dataset, ver=ver)

    set_global_v(x_train.shape[0])

    model.compile(loss=spl_loss, optimizer=optimizer, metrics=['accuracy'])

    if useCLR:
        clr = CyclicLRCustom(mode='triangular', base_lr=0.0001, max_lr=0.1, step_size=200)

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # get the batches
    batch_num = int(x_train.shape[0] / batch_size)
    train_batch_x = [x_train[batch_size * i: batch_size * (i+1)] for i in range(batch_num)]
    train_batch_y = [y_train[batch_size * i: batch_size * (i+1)] for i in range(batch_num)]
    batch_index = [[j for j in range(batch_size * i, batch_size * (i+1))] for i in range(batch_num)]

    for epoch in range(epochs):
        print('[epoch {}] sum v: {}'.format(epoch, np.sum(v)))
        for bat_x, bat_y, index in zip(train_batch_x, train_batch_y, batch_index):
            set_current_batch_idx(np.array(index))
            loss, acc = model.train_on_batch(bat_x, bat_y)
            if useCLR:
                clr.on_batch_end(model)
        increase_threshold()
        train_loss, train_acc = model.evaluate(x_train, y_train, batch_size=batch_size)
        test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        
        print('train acc: {:.4f}, test acc: {:.4f}'.format(train_acc, test_acc))
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

        np.save(os.path.join(model_weights_dir, 'spl_train_acc.npy'), np.array(train_acc_list))
        np.save(os.path.join(model_weights_dir, 'spl_train_loss.npy'), np.array(train_loss_list))
        np.save(os.path.join(model_weights_dir, 'spl_test_acc.npy'), np.array(test_acc_list))
        np.save(os.path.join(model_weights_dir, 'spl_test_loss.npy'), np.array(test_loss_list))

        model.save_weights(trained_weights_path)
    
