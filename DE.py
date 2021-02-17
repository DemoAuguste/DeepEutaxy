from models import *
from utils import *
from settings import *
from CLR import CyclicLR

import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str) # resnet20, resnet32, lenet5, cnn
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-v', '--version', type=str, default='1')
    parser.add_argument('-c', '--clr', type=int, default=0)

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


    x_train, y_train, x_test, y_test = get_dataset(dataset, ver=ver)

    if dataset == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10

    batch_num = int(x_train.shape[0] / batch_size)
    train_batch_x = [x_train[batch_size * i: batch_size * (i+1)] for i in range(batch_num)]
    train_batch_y = [y_train[batch_size * i: batch_size * (i+1)] for i in range(batch_num)]
    batch_index = [i for i in range(batch_num)]

    model = get_model(model_name, compiled=True)        

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    trained_weights_path = os.path.join(model_weights_dir, 'DE_trained.h5')

    if dataset == 'mnist':
        total_iter_num = 40
        epochs_num = 5
    elif dataset == 'cifar10':
        total_iter_num = 20
        epochs_num = 10

    if useCLR:
        clr = CyclicLR(mode='triangular', base_lr=0.0001, max_lr=0.1, step_size=200)

    print(total_iter_num, epochs_num)
    for i in range(total_iter_num):
        batch_index = get_batch_similarity(model, train_batch_x, train_batch_y, batch_index)
        
        temp_train_x = x_train
        temp_train_y = y_train
        print('[iteration {}]'.format(i))

        for index in batch_index:
            temp_train_x[batch_size * i: batch_size * (i+1)] = train_batch_x[index]
            temp_train_y[batch_size * i: batch_size * (i+1)] = train_batch_y[index]

        if useCLR:
            hist = model.fit(x=temp_train_x, y=temp_train_y, 
                                batch_size=1000, 
                                epochs=epochs_num, 
                                validation_data=(x_test, y_test),
                                callbacks=[clr],
                                shuffle=False)
        else:
            hist = model.fit(x=temp_train_x, y=temp_train_y, 
                                batch_size=1000, 
                                epochs=epochs_num, 
                                validation_data=(x_test, y_test), 
                                shuffle=False)


        print('[epoch {}] train acc: {}, test acc: {}'.format(i, hist.history['accuracy'], hist.history['val_accuracy']))

        train_acc_list.extend(hist.history['accuracy'])
        train_loss_list.extend(hist.history['loss'])
        test_acc_list.extend(hist.history['val_accuracy'])
        test_loss_list.extend(hist.history['val_loss'])
            

        model.save_weights(trained_weights_path)

    
        np.save(os.path.join(model_weights_dir, 'de_train_acc.npy'), np.array(train_acc_list))
        np.save(os.path.join(model_weights_dir, 'de_train_loss.npy'), np.array(train_loss_list))
        np.save(os.path.join(model_weights_dir, 'de_test_acc.npy'), np.array(test_acc_list))
        np.save(os.path.join(model_weights_dir, 'de_test_loss.npy'), np.array(test_loss_list))

