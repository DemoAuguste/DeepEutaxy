from models import *
import tensorflow as tf

def get_model(model_name, compiled=True, num_classes=10):
    if compiled:
        if model_name == 'resnet20':
            model = get_resnet_20(num_classes)
        elif model_name == 'resnet32':
            model = get_resnet_32(num_classes)
        elif model_name == 'cnn':
            model = model1()
        elif model_name == 'lenet5':
            model = lenet5()
        elif model_name == 'mobilenetv2':
            model = get_mobilenet_v2(num_classes)
        return model
    else:
        if model_name == 'resnet20':
            model, optimizer = get_resnet_20_uncompiled(num_classes)
        elif model_name == 'resnet32':
            model, optimizer = get_resnet_32_uncompiled(num_classes)
        elif model_name == 'cnn':
            model, optimizer = model1_uncompiled()
        elif model_name == 'lenet5':
            model, optimizer = lenet5_uncompiled()
        elif model_name == 'mobilenetv2':
            model, optimizer = get_mobilenet_v2_uncompiled(num_classes)
        return model, optimizer
        