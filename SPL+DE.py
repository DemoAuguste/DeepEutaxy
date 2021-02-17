import tensorflow as tf
import numpy as np
import argparse
import os
import keras
from keras import backend as K
from models import *
from utils import *
from settings import *

tf.config.experimental_run_functions_eagerly(True) 
# exit()
v = None
threshold = -100
growing_factor = 1.3
ce_loss = tf.keras.losses.CategoricalCrossentropy()
current_batch_index = None
batch_num = None
DE_batch_index = None 
iter_count = 0


class MyCallBack(keras.callbacks.Callback):
    def on_batch_begin(self, batch, logs):
        global iter_count, DE_batch_index
        temp_index = [j for j in range(batch_size * DE_batch_index[iter_count], batch_size * (DE_batch_index[iter_count] + 1))]
        set_current_batch_idx(np.array(temp_index, dtype=int))
        return 

    def on_batch_end(self, batch, logs):
        global iter_count, DE_batch_index
        iter_count += 1
        return
    
    def on_epoch_end(self, epoch, logs):
        global iter_count, v
        iter_count = 0
        increase_threshold()
        return


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
    v = np.ones(n_samples, dtype=np.int) 

def set_current_batch_idx(value):
    global current_batch_index
    current_batch_index = value
    return current_batch_index

def increase_threshold():
    global threshold, growing_factor
    threshold *= growing_factor

def get_batch_similarity_SPL_DE(model, batch_x, batch_y, batch_index):
    global current_batch_index
    curr_weights = model.trainable_weights
    sim_list = []
    for index in batch_index:
        bat_x = batch_x[index]
        bat_y = batch_y[index]

        temp_index = [j for j in range(batch_size * index, batch_size * (index + 1))]
        set_current_batch_idx(np.array(temp_index, dtype=int))

        grads = get_weight_grad(model, bat_x, bat_y)

        sim = get_weights_similarity(curr_weights, grads)

        sim_list.append(sim)
    indexed_sim_list = [(item[0], item[1]) for item in zip(batch_index, sim_list)]
    sort_list = sorted(indexed_sim_list, key=lambda x: x[1], reverse=False)
    reordered_index = [item[0] for item in sort_list]

    return reordered_index



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='resnet20')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10')
    parser.add_argument('-v', '--version', type=str, default='1')
    parser.add_argument('-u', '--eutaxy', type=int, default=0)

    args = parser.parse_args()

    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    ver = args.version
    isDE = args.eutaxy

    model_weights_dir = os.path.join(weights_dir, model_name)
    model_weights_dir = os.path.join(model_weights_dir, dataset)
    model_weights_dir = os.path.join(model_weights_dir, ver)

    if not os.path.exists(model_weights_dir):
        os.makedirs(model_weights_dir)
    if isDE:
        trained_weights_path = os.path.join(model_weights_dir, 'spl_de_trained.h5')
    else:
        trained_weights_path = os.path.join(model_weights_dir, 'spl_new_trained.h5')

    model, optimizer = get_model(model_name, False)
    x_train, y_train, x_test, y_test = get_dataset(dataset, ver=ver)

    # load weights.
    load_weights_path = os.path.join(model_weights_dir, 'spl_trained.h5')
    model.load_weights(load_weights_path)


    set_global_v(x_train.shape[0])

    # compile the model with SPL loss
    
    model.compile(loss=spl_loss, optimizer=optimizer, metrics=['accuracy'])

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # get the batches
    batch_num = int(x_train.shape[0] / batch_size)
    train_batch_x = [x_train[batch_size * i: batch_size * (i+1)] for i in range(batch_num)]
    train_batch_y = [y_train[batch_size * i: batch_size * (i+1)] for i in range(batch_num)]
    batch_index = [[j for j in range(batch_size * i, batch_size * (i+1))] for i in range(batch_num)]

    DE_batch_index = [i for i in range(batch_num)]

    if dataset == 'mnist':
        total_iter_num = 40
        epochs_num = 5
    elif dataset == 'cifar10':
        total_iter_num = 10
        epochs_num = 5
    
    print(total_iter_num, epochs_num)

    cb = MyCallBack()

    for i in range(total_iter_num):
        print('[iteration {}] sum v: {}'.format(i, np.sum(v)))
        if isDE:
            DE_batch_index = get_batch_similarity_SPL_DE(model, train_batch_x, train_batch_y, DE_batch_index)
            print(DE_batch_index)

        # for epoch in range(epochs_num):
        temp_train_x = x_train
        temp_train_y = y_train

        if isDE:
            for item, index in enumerate(DE_batch_index):
                temp_train_x[batch_size * item: batch_size * (item+1)] = train_batch_x[index]
                temp_train_y[batch_size * item: batch_size * (item+1)] = train_batch_y[index]

        hist = model.fit(x=temp_train_x, y=temp_train_y, batch_size=1000, 
                            epochs=epochs_num, 
                            validation_data=(x_test, y_test), 
                            shuffle=False,
                            callbacks=[cb])
        
        train_acc_list.extend(hist.history['accuracy'])
        train_loss_list.extend(hist.history['loss'])
        test_acc_list.extend(hist.history['val_accuracy'])
        test_loss_list.extend(hist.history['val_loss'])

        if isDE:
            np.save(os.path.join(model_weights_dir, 'spl_de_train_acc.npy'), np.array(train_acc_list))
            np.save(os.path.join(model_weights_dir, 'spl_de_train_loss.npy'), np.array(train_loss_list))
            np.save(os.path.join(model_weights_dir, 'spl_de_test_acc.npy'), np.array(test_acc_list))
            np.save(os.path.join(model_weights_dir, 'spl_de_test_loss.npy'), np.array(test_loss_list))
        else:
            np.save(os.path.join(model_weights_dir, 'spl_train_acc.npy'), np.array(train_acc_list))
            np.save(os.path.join(model_weights_dir, 'spl_train_loss.npy'), np.array(train_loss_list))
            np.save(os.path.join(model_weights_dir, 'spl_test_acc.npy'), np.array(test_acc_list))
            np.save(os.path.join(model_weights_dir, 'spl_test_loss.npy'), np.array(test_loss_list))
    

        model.save_weights(trained_weights_path)