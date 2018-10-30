from keras import optimizers, losses
from keras import metrics
import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import csv
import copy

n_vector_dimension = 5727
n_layer1_units = 1024
n_layer1_units = 1024
n_layer2_units = 1024
n_layer3_units = 512
n_layer4_units = 512
n_layer5_units = 512
n_output_dimension = 1

model = Sequential()

model.add(Dense(n_layer1_units, input_dim=n_vector_dimension))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(n_layer2_units))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(n_layer3_units))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(n_layer4_units))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(n_layer5_units))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(n_output_dimension))

optimizer = optimizers.Adam()

def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = K.abs(error) < clip_delta
    
    squared_loss = 0.5 * K.square(error)
    linear_loss = clip_delta * (K.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)

def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
    return K.mean(huber_loss(y_true, y_pred, clip_delta))

model.compile(
    optimizer=optimizer,
    loss=huber_loss_mean
)

train_set_path = './train.csv'
test_set_path = './test.csv'

train_set_lines = 33213
batch_size = 64

def train_batch_generator(path, batch_size):
    lines_of_batch = []
    while 1:
        csv_file = csv.reader(open(path))
        for line in csv_file:
            if len(lines_of_batch) < batch_size:
                lines_of_batch.append(line)
            else:
                return_batch = copy.deepcopy(lines_of_batch)
                return_array = np.array(return_batch, dtype=np.float32)
                return_array_X = return_array[:, :n_vector_dimension]
                return_array_y = return_array[:, -1]
                return_array_y = return_array_y.reshape([return_array_y.shape[0], 1])
                lines_of_batch = []
                yield (return_array_X, return_array_y)

test_set = np.genfromtxt(test_set_path, delimiter=',', dtype=np.float32)
test_X = test_set[:, :n_vector_dimension]
test_y = test_set[:, -1]
test_y = test_y.reshape(test_y.shape[0], 1)

test_X_sample = test_X[:200, :]
test_y_sample = test_y[:200, :]

steps_per_epoch = train_set_lines // batch_size
train_generator = train_batch_generator(train_set_path, batch_size)

# model.fit_generator(
#         train_generator,
#         epochs=1,
#         validation_data=(test_X_sample, test_y_sample),
#         steps_per_epoch=steps_per_epoch,
#         verbose=1
#     )
# model.save("reg_baseline.h5")

epochs = 250

# for i in range(epochs):
#     model.fit_generator(train_generator, epochs=1, validation_data=(test_X_sample, test_y_sample), steps_per_epoch=steps_per_epoch, verbose=1)
#     model.save(f'reg_epoch_{i}.h5')

def validate_acc(model_path, validation_X, validation_y):
    mappings = { 'huber_loss_mean': huber_loss_mean }
    model = load_model(model_path, custom_objects=mappings)
    validation_y_pred = model.predict(validation_X)
    validation_y_pred = np.rint(validation_y_pred)
    validation_y_diff = validation_y_pred - validation_y
    element_count = validation_y.size
    nonzero_count = np.count_nonzero(validation_y_diff)
    acc = (element_count - nonzero_count) / element_count
    K.clear_session()
    return acc

validation_set_path = "./validation.csv"
validation_set = np.genfromtxt(validation_set_path, delimiter=',', dtype=np.float32)
validation_set_X = validation_set[:, :n_vector_dimension]
validation_set_y = validation_set[:, -1]
validation_set_y = validation_set_y.reshape(validation_set_y.shape[0], 1) 

accs = []
for i in range(epochs):
    acc = validate_acc(f"reg_epoch_{i}.h5", validation_set_X, validation_set_y)
    print(f"Model of epoch {i}, Accuracy: {acc}")
    accs.append(acc)

plt.plot(range(epochs), accs)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (on validation set)")
plt.show()