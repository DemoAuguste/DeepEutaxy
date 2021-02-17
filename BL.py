from models import *
from utils import *
from settings import *
from CLR import CyclicLRCustom

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

    x_train, y_train, x_test, y_test = get_dataset(dataset)

    if dataset == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10

    batch_num = int(x_train.shape[0] / batch_size)
    train_batch_x = [x_train[batch_size * i: batch_size * (i+1)] for i in range(batch_num)]
    train_batch_y = [y_train[batch_size * i: batch_size * (i+1)] for i in range(batch_num)]
    batch_index = [i for i in range(batch_num)]

    model = get_model(model_name, compiled=True)

    if useCLR:
        clr = CyclicLRCustom(mode='triangular', base_lr=0.0001, max_lr=0.1, step_size=200)
        

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # baseline.
    trained_weights_path = os.path.join(model_weights_dir, 'BL_trained.h5')
    for epoch in range(epochs):
        for bat_x, bat_y in zip(train_batch_x, train_batch_y):
            loss, acc = model.train_on_batch(bat_x, bat_y)
            if useCLR:
                clr.on_batch_end(model)
        train_loss, train_acc = model.evaluate(x_train, y_train)
        test_loss, test_acc = model.evaluate(x_test, y_test)

        print('[epoch {}] train acc: {:.4f}, test acc: {:.4f}'.format(epoch, train_acc, test_acc))

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

        model.save_weights(trained_weights_path)

    np.save(os.path.join(model_weights_dir, 'base_train_acc.npy'), np.array(train_acc_list))
    np.save(os.path.join(model_weights_dir, 'base_train_loss.npy'), np.array(train_loss_list))
    np.save(os.path.join(model_weights_dir, 'base_test_acc.npy'), np.array(test_acc_list))
    np.save(os.path.join(model_weights_dir, 'base_test_loss.npy'), np.array(test_loss_list))
        
  