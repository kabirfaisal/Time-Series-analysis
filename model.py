import time
start_time = time.time()

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset set
dataset = pd.read_csv('dataset.csv')
dataset_train = dataset[0:10000]
dataset_test = dataset[10001:13000]
training_set = dataset_train.iloc[:, 1:2].values
print(dataset_train.head())


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
#training_set_scaled.head()

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 10000):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train.shape)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN
#import keras
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#from keras.layers import Activation
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
#regressor.add(Activation('relu'))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
#regressor.add(Activation('relu'))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
#regressor.add(Activation('relu'))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
#regressor.add(Activation('relu'))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))
#regressor.add(Activation('sigmoid'))
regressor.summary()
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 1, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real price 
real_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted price
dataset_total = pd.concat((dataset_train['e5'], dataset_test['e5']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 3059):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_price = regressor.predict(X_test)
predicted_price = sc.inverse_transform(predicted_price)

# Final evaluation of the model
scores = regressor.evaluate(X_test, predicted_price, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Visualising the results
plt.plot(real_price, color = 'red', label = 'Fuel Price for E5')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Fuel Price for E5')
plt.title('Fuel Price for E5 Prediction')
plt.xlabel('Time')
plt.ylabel('Fuel Price for E5')
plt.legend()
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))