import os

working_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(working_dir, 'data')
weights_dir = os.path.join(working_dir, 'weights')

batch_size = 1000
epochs = 200

iter_count = 10
