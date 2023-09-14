import numpy as np
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

# Load data from the CSV file
data = pd.read_csv('linear_reg_nonzero.csv')

# Assuming that the CSV file has columns named 'independent', 'dependent1', and 'dependent2'
X = data['d'].values
Y1 = -data['h'].values
Y2 = data['w'].values

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adjust the learning rate

# Define and compile your linear regression model as mentioned before
model = Sequential()
model.add(Dense(32, input_dim=1, kernel_initializer='normal', activation='relu'))  # Add the first hidden layer
model.add(Dense(16, activation='relu'))  # Add the second hidden layer
model.add(Dense(1, activation='linear'))  # Output layer with 2 units
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Define the ModelCheckpoint callback to save the model when validation accuracy improves
checkpoint = ModelCheckpoint(
    'best_model.h5',  # File name to save the best model
    monitor='val_loss',  # Metric to monitor (e.g., validation loss)
    save_best_only=True,  # Save only the best model
    mode='min',  # 'min' for loss, 'max' for accuracy, 'auto' for automatic
    verbose=1  # Verbosity level
)

# Train the model with the callback
model.fit(
    X, Y1,
    epochs=1000,
    batch_size=64,
    validation_split=0.05,  # Specify the validation split
    callbacks=[checkpoint]  # Pass the checkpoint callback
)

model.save('final_model.h5')

model2 = Sequential()
model2.add(Dense(64, input_dim=1, kernel_initializer='normal', activation='relu'))  # Add the first hidden layer
model2.add(Dense(32, activation='relu'))  # Add the second hidden layer
model2.add(Dense(1, activation='linear'))  # Output layer with 2 units
model2.compile(loss='mean_squared_error', optimizer=optimizer)

# Define the ModelCheckpoint callback to save the model when validation accuracy improves
checkpoint2 = ModelCheckpoint(
    'best_model2.h5',  # File name to save the best model
    monitor='val_loss',  # Metric to monitor (e.g., validation loss)
    save_best_only=True,  # Save only the best model
    mode='min',  # 'min' for loss, 'max' for accuracy, 'auto' for automatic
    verbose=1  # Verbosity level
)

model2.fit(
    X, Y2,
    epochs=1000,
    batch_size=256,
    validation_split=0.05,  # Specify the validation split
    callbacks=[checkpoint2]  # Pass the checkpoint callback
)

model2.save('final_model2.h5')