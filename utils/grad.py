from scipy.spatial import distance
import copy
from settings import batch_size
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.mixed_precision.experimental import (
    loss_scale_optimizer as lso)

# def get_weight_grad(model, inputs, outputs):
#     """ Gets gradient of model for given inputs and outputs for all weights"""
#     grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
#     symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
#     f = K.function(symb_inputs, grads)
#     x, y, sample_weight = model._standardize_user_data(inputs, outputs)
#     output_grad = f(x + y + sample_weight)
#     return output_grad


def get_weight_grad(model, x, y, sample_weight=None, learning_phase=0):
    def _process_input_data(x, y, sample_weight, model):
        iterator = data_adapter.single_batch_iterator(model.distribute_strategy,
                                                      x, y, sample_weight,
                                                      class_weight=None)
        data = next(iterator)
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        return x, y, sample_weight

    def _clip_scale_grads(strategy, tape, optimizer, loss, params):
        with tape:
            if isinstance(optimizer, lso.LossScaleOptimizer):
                loss = optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, params)

        aggregate_grads_outside_optimizer = (
            optimizer._HAS_AGGREGATE_GRAD and not isinstance(
                strategy.extended,
                parameter_server_strategy.ParameterServerStrategyExtended))

        if aggregate_grads_outside_optimizer:
            gradients = optimizer._aggregate_gradients(zip(gradients, params))
        if isinstance(optimizer, lso.LossScaleOptimizer):
            gradients = optimizer.get_unscaled_gradients(gradients)

        gradients = optimizer._clip_gradients(gradients)
        return gradients

    x, y, sample_weight = _process_input_data(x, y, sample_weight, model)

    with tf.GradientTape() as tape:
        y_pred = model(x, training=bool(learning_phase))
        loss = tf.keras.losses.CategoricalCrossentropy()(y, y_pred)
        
        # loss = model.compiled_loss(y, y_pred, sample_weight,
        #                            regularization_losses=model.losses)

    gradients = _clip_scale_grads(model.distribute_strategy, tape,
                                  model.optimizer, loss, model.trainable_weights)
    gradients = K.batch_get_value(gradients)
    return gradients





def get_layer_output_grad(model, inputs, outputs, layer=-1):
    """ Gets gradient a layer output for given inputs and outputs"""
    grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad

def get_grad_norm(grads):
    _sum = 0.0
    for grad in grads:
        _sum += np.linalg.norm(grad)
    return _sum


def get_batch_similarity(model, batch_x, batch_y, batch_index):
    curr_weights = model.trainable_weights
    sim_list = []
    for index in batch_index:
        bat_x = batch_x[index]
        bat_y = batch_y[index]

        # after_model = copy.deepcopy(model)
        # after_model.train_on_batch(bat_x, bat_y)
        # after_weights = get_model_weights(model)
        grads = get_weight_grad(model, bat_x, bat_y)
        # grads_norm = get_grad_norm(grads)
        sim = get_weights_similarity(curr_weights, grads)
        # sim = get_weights_similarity(curr_weights, after_weights)
        sim_list.append(sim)
    indexed_sim_list = [(item[0], item[1]) for item in zip(batch_index, sim_list)]
    sort_list = sorted(indexed_sim_list, key=lambda x: x[1], reverse=False)
    reordered_index = [item[0] for item in sort_list]

    return reordered_index
    
    
def get_weights_similarity(w1, w2):
    sum_sim = 0
    for a, b in zip(w1, w2):
        temp_a = a.numpy()
        temp_a[np.isnan(temp_a)] = 0
        # temp_b = b.numpy()
        # temp_b[np.isnan(temp_b)] = 0
        ret = distance.cosine(temp_a.flatten(), b.flatten())
        if np.isnan(ret):
            # print('not valid')
            continue 
        sum_sim += ret
    return sum_sim


def get_model_weights(model):
    weights_list = []
    for layer in model.layers:
        if len(layer.get_weights()) == 0:
            continue
        weights_list.append(layer.get_weights()[0])
    return weights_list
