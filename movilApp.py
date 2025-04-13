import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

datos = {
    'idRuta': [101,102,103,104,105,106,107,108],
    'horaDia': ['07:00', '18:00', '12:00', '15:00', '06:30', '17:40', '10:00', '20:00'],
    'diaSemana': ['lunes', 'viernes', 'domingo', 'miercoles', 'martes', 'jueves', 'sábado', 'lunes'], 
    'pasajeros' : [ 34, 58, 20, 45, 30, 50, 25, 60],
    'retardoMin' : [3, 10, 0, 5, 2, 7, 1, 12],
    'clima': ['soleado', 'lluvia', 'nublado', 'lluvia', 'soleado', 'nublado', 'soleado', 'lluvia'],
    'trafico': ['medio', 'alto', 'bajo', 'alto', 'medio', 'alto', 'bajo', 'alto'],
    'tiempoLlegada' : [15, 35, 10, 25, 12, 30, 11, 40]
}


df_original = pd.DataFrame(datos)

df = df_original.copy()

#codificamos variables de categoria
label_encoders = {} #Variable que guardara los encoders
for col in ['horaDia', 'diaSemana', 'clima', 'trafico']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le #Guardamos encoders
    
#Separamos variables 
x = df.drop('tiempoLlegada', axis=1)
y = df['tiempoLlegada']

#Dividimos datos.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#Entrenamos modelos.
modelo = DecisionTreeRegressor(max_depth=4, random_state=42)
modelo.fit(X_train, y_train)

#Evaluamos los modelos.
y_pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

#Obtenemos indices
indices_test = X_test.index

#Creamos tabla.
tabla_resultado = df_original.loc[indices_test].copy()
tabla_resultado['Predicción'] = y_pred

print("Tabla con datos originales y predicciones: ")
print(tabla_resultado.to_string(index=False))

print("\nError Absoluto Medio:", mae)
