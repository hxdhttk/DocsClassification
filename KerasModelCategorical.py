from keras import optimizers, losses
from keras import metrics
import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
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
n_output_dimension = 59

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
model.add(Activation('softmax'))

optimizer = optimizers.Adam()

model.compile(
    optimizer=optimizer,
    loss=losses.categorical_crossentropy,
    metrics=[metrics.categorical_accuracy]
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
                return_array_y = np_utils.to_categorical(return_array_y, num_classes=n_output_dimension)
                lines_of_batch = []
                yield (return_array_X, return_array_y)

test_set = np.genfromtxt(test_set_path, delimiter=',', dtype=np.float32)
test_X = test_set[:, :n_vector_dimension]
test_y = test_set[:, -1]
test_y = test_y.reshape(test_y.shape[0], 1)
test_y = np_utils.to_categorical(test_y, num_classes=n_output_dimension)

test_X_sample = test_X[:500, :]
test_y_sample = test_y[:500, :]

steps_per_epoch = train_set_lines // batch_size
train_generator = train_batch_generator(train_set_path, batch_size)

# model.fit_generator(
#         train_generator,
#         epochs=1,
#         validation_data=(test_X_sample, test_y_sample),
#         steps_per_epoch=steps_per_epoch,
#         verbose=1
#     )
# model.save("cat_baseline.h5")

epochs = 250

# for i in range(epochs):
#     model.fit_generator(
#         train_generator, 
#         epochs=1, 
#         validation_data=(test_X_sample, test_y_sample), 
#         steps_per_epoch=steps_per_epoch, 
#         verbose=1
#     )
#     model.save(f'cat_epoch_{i}.h5')


validation_set_path = "./validation.csv"
validation_set = np.genfromtxt(validation_set_path, delimiter=',', dtype=np.float32)
validation_set_X = validation_set[:, :n_vector_dimension]
validation_set_y = validation_set[:, -1]
validation_set_y = validation_set_y.reshape(validation_set_y.shape[0], 1)
validation_set_y = np_utils.to_categorical(validation_set_y, num_classes=n_output_dimension)

accs = []
for i in range(epochs):
    model = load_model(f"cat_epoch_{i}.h5")
    scores = model.evaluate(x=validation_set_X, y=validation_set_y)
    acc = scores[1]
    print(f"Model of epoch {i}, Accuracy: {acc}")
    accs.append(acc)
    K.clear_session()

plt.plot(range(epochs), accs)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (on validation set)")
plt.show()
