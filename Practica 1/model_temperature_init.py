# Muestra los valores más anómalos del fichero csv suministrado

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

# Cargar los datos
df = pd.read_csv("datos.csv",index_col=0,parse_dates=True)

print(df)
print(df.shape) # (7267,1) (según el fichero cambiará el número de filas)

df.plot()
plt.show()

# Crear las ventanas temporales
# Lo que se predice (y) es el "siguiente" valor de la secuencia (sin normalizar)
# pasando la ventana actual que tenemos.
#Las salidas no van a estar normalizadas(array y) pero las entradas sí (Array X)


#Con el código proporcionado en el Campus Virtual vamos a reutilizar la funcion
#split_sequence, el cual nos devuelve un array X con todas las ventanas de tamanio
#n_steps y un array Y con el valor inmediatamente posterior.
#Ej: datos = [3,4,5,6,7] n_steps = 2 -> X = [[3,4],[4,5],[5,6]] y = [5,6,7]
# split a univariate sequence
#Además, vamos a modificar ligeramente la función para que solo el array X tenga los
#valores normalizados
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

n_steps = 3
X, y = split_sequence(df['value'], n_steps)

# Normalizar los datos
#gastar cuidado con nuevos datos para predecir, aplicar el mismo escalador

scaler = MinMaxScaler() # Usar esta función
df_scaled =  scaler.fit_transform(X)#ajusta datos entre 0 y 1

# Dividir los datos en entrenamiento y prueba es lo habitual (80-20 o 70-30)
# Aunque en este caso, vamos a querer luego detectar anomalías en todos los datos

#Entrenando todos los datos:

# Redimensionar los datos para la RNN
# LSTM espera 3 dimensiones: número muestras, pasos temporales (cada ventana tamaño 10), número features (1)
# P.ej: (5805,10,1)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# Crear la RNN
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features))) #relu -> rectificación lineal
model.add(Dense(1)) 
model.compile(optimizer='adam', loss='mse')

# Entrenar la RNN
# fit model
model.fit(X, y, epochs=200)


# demonstrate prediction
x_input = np.array([69.88, 71.22, 70.87])
x_input = x_input.reshape((1, n_steps, n_features)) #solo queremos una muestra
yhat = model.predict(x_input, verbose=0) #Usamos esto para conseguir todas las muestras predichas
print(yhat)


#Entrenando 70% de los datos:

# Redimensionar los datos para la RNN
# LSTM espera 3 dimensiones: número muestras, pasos temporales (cada ventana tamaño 10), número features (1)
# P.ej: (5805,10,1)

# Crear la RNN

# Entrenar la RNN


#termina la primera parte

#Gráfica para hacer al final

#Creamos un array con todas las predicciones
y_pred = np.array()

# Un posible criterio de anomalía
#  Calcular el error absoluto medio (MAE)
# mae = tf.keras.metrics.mean_absolute_error(y_test, y_pred).numpy() Cuidado, no funciona bien

# Otro posible criterio: uso de percentiles

# Mostrar las fechas de las anomalías (dentro de las ventanas)

# los valores de anomalias se refieren a las ventanas, no a valores específicos
