import os
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.initializers import Orthogonal
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

actions = ['a', 'b', 'c', 'fuck']

data = np.concatenate([
    np.load('dataset/seq_a.npy'),
    np.load('dataset/seq_b.npy'),
    np.load('dataset/seq_c.npy'),
    np.load('dataset/seq_fuck.npy'),
    # np.load('dataset/seq_e.npy')
], axis=0)

x_data = data[:, :, :-1]
labels = data[:, 0, -1]
y_data = to_categorical(labels, num_classes=4)
x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2024)

# 초기화 방법
initializers = Orthogonal(gain=1.0, seed=None)
dr = 0.3

model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3], kernel_initializer=initializers),
    Dropout(dr),
    Dense(32, activation='relu', kernel_initializer=initializers),
    Dropout(dr),
    Dense(len(actions), activation='softmax', kernel_initializer=initializers)
])

lr = 0.001
model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

folder_path = 'C:/Users/user/Downloads/Mediapipe-LSTM/'

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[
        ModelCheckpoint(folder_path + 'models/model2.keras', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(factor=0.5, patience=50, verbose=1, mode='auto')
    ]
)
