from keras.models import Model, load_model
from keras.layers import Dense, Input, Activation, BatchNormalization, GlobalAveragePooling1D, Conv1D, AveragePooling1D
from keras.utils import to_categorical
from keras import optimizers
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

n_input_shape = None
n_kernel_size = 7
n_pooling_size = 2
n_filters_1 = 8
n_filters_2 = 16
n_filters_3 = 32
n_labels = 59

# Input
sequence_input = Input(shape=(n_input_shape, 1))

# Conv 1, 2 & 3
x = Conv1D(n_filters_1, n_kernel_size, padding='same')(sequence_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv1D(n_filters_1, n_kernel_size, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv1D(n_filters_1, n_kernel_size, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = AveragePooling1D(n_pooling_size)(x)

# Conv 4, 5, & 6
x = Conv1D(n_filters_2, n_kernel_size, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv1D(n_filters_2, n_kernel_size, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv1D(n_filters_2, n_kernel_size, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = AveragePooling1D(n_pooling_size)(x)

# Conv 7, 8 & 9
x = Conv1D(n_filters_3, n_kernel_size, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv1D(n_filters_3, n_kernel_size, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv1D(n_filters_3, n_kernel_size, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = GlobalAveragePooling1D()(x)

# Output
preds = Dense(n_labels, activation='softmax')(x)

model = Model(sequence_input, preds)

adam = optimizers.Adam()

model.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['acc']
)

# train_X_path = "./train_X.csv"
# train_y_path = "./train_y.csv"

# epochs = 500

# for i in range(epochs):
#     train_X = open(train_X_path)
#     train_y = open(train_y_path)
#     for X, y in zip(train_X, train_y):
#         X_sample = np.array(X.split(','), dtype=np.float32)
#         X_sample = X_sample.reshape([1, X_sample.shape[0], 1])
#         y_sample = to_categorical(int(y), num_classes=n_labels)
#         y_sample = np.array([y_sample])
#         model.fit(X_sample, y_sample, batch_size=1, epochs=1, shuffle=False)
#     model.save(f"cat_epoch_{i}.h5")
#     model.reset_states()

model_range = 300

def validation_acc(model_path, validation_X, validation_y):
    model = load_model(model_path)
    results = []
    for X, y in zip(validation_X, validation_y):
        X_sample = np.array(X.split(','), dtype=np.float32)
        X_sample = X_sample.reshape([1, X_sample.shape[0], 1])
        y_pred = model.predict(X_sample)
        y_pred_index = np.argmax(y_pred)
        diff = y_pred_index - int(y)
        results.append(diff)
    element_count = len(results)
    correct_pred_count = len(list(filter(lambda x: x == 0, results)))
    K.clear_session()
    return correct_pred_count / element_count

accs = []
for i in range(model_range):
    validation_X = open("./test_X.csv")
    validation_y = open("./test_y.csv")
    acc = validation_acc(f"cat_epoch_{i}.h5", validation_X, validation_y)
    print(f"Model of Epoch {i}, Accuracy: {acc}")
    accs.append(acc)
    validation_X.close()
    validation_y.close()

plt.plot(range(model_range), accs)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()