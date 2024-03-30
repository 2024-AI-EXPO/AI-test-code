import os
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.initializers import Orthogonal
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

actions = ['f', 'g', 'h', 'i']
path = 'dataset'
data = np.concatenate([np.load(path + f'/seq_{action}.npy') for action in actions], axis=0)

x_data = data[:, :, :-1]
labels = data[:, 0, -1]
y_data = to_categorical(labels, num_classes=len(actions))
x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2024)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
# 초기화 방법
initializers = Orthogonal(gain=1.0, seed=None)
dr = 0.3

model = Sequential([
    Input(shape=x_train.shape[1:3]),
    LSTM(64, activation='tanh', kernel_initializer=initializers),
    Dropout(dr),
    Dense(32, activation='tanh', kernel_initializer=initializers),
    Dropout(dr),
    Dense(len(actions), activation='softmax', kernel_initializer=initializers)
])

lr = 0.0001
model.compile(optimizer=SGD(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

folder_path = 'C:/Users/user/Downloads/AI-test-code/Mediapipe-Test/'

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=8,
    callbacks=[
        ModelCheckpoint(folder_path + 'testmodel/test5.keras', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(factor=0.5, patience=50, verbose=1, mode='auto')
    ]
)

# 확인 코드
fig, loss_ax = plt.subplots(figsize=(12, 5))
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train_loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val_loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper right')

acc_ax.plot(history.history['accuracy'], 'b', label='train_acc')
acc_ax.plot(history.history['val_accuracy'], 'g', label='val_acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='center right')
plt.ylim(0.0, 1.1)
plt.show()
