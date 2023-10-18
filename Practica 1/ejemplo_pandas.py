# https://www.geeksforgeeks.org/python-pandas-dataframe/
import pandas as pd #para no poner pandas podemos poner motes
import matplotlib.pyplot as plt

# making data frame from csv file
df = pd.read_csv("https://media.geeksforgeeks.org/wp-content/uploads/nba.csv", index_col ="Name") #df nos acelera el dataframe

print(df.head())

print(df.describe())

df['Weight'].plot(label='weight') #librería matplotlib
plt.legend()
plt.show()

input('select two columns')
df2=df[['Team', 'Number']] #para seleccionar columnas que necesitamos
print(df2)

input('rows 0 to 2, all columns')
df3 = df.iloc[0:3, :] #coge la fila de la 0 a la 2 y coge todas las columnas, útil para 70% entrenamiento resto validaciónls p
print(df3.head())

input('retrieving row by loc method')
first = df.loc["Avery Bradley"]
second = df.loc["R.J. Hunter"]
  
print(first, "\n\n\n", second)

input('iterating rows and specific column')
for i in range(len(df)):
  print(df.iat[i,0])

input('sort')
print(df.sort_values(by='Name', ascending = True))

input('filtering')
print(df[df["Age"] > 30])

input('New column')
df['FromKentucky'] = df["College"] == "Kentucky" #crea columna de booleans, útil para saber si las muestras tienen anomalías
print(df)

input('statistical')
res = df[["Age", "Salary"]].mean()
print(res)

print(df[["Age"]].min())
print(df[["Age"]].max())
