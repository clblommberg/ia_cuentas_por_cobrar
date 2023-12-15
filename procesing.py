import pandas as pd
import numpy as np
from openpyxl import Workbook 
import os
import datetime 

import matplotlib.pyplot as plt
import seaborn as sns

# estas son las clases para sustitutición con sklearn
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# dividir dataset
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier

# one hot encoding con feature-engine
from feature_engine.encoding import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier

# GroupKFold en un pipeline con StandardScaler y SVC

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold

from sklearn.model_selection import GridSearchCV

import os
# Obtener la ruta del directorio actual
ruta_actual = os.getcwd()
print(ruta_actual)
# Crear una ruta relativa basada en el directorio actual
ruta_relativa = os.path.join(ruta_actual, 'data/raw/data.xlsx')

print(ruta_relativa)

#Leer el archivo con pandas 
#working_dir = "accountant_receivable/"

df_mayor = pd.read_excel(ruta_relativa, sheet_name="ldiario")
df_clients = pd.read_excel(ruta_relativa, sheet_name="clientes")
df_sales = pd.read_excel(ruta_relativa, sheet_name="ventas")
df_enterprise= pd.read_excel(ruta_relativa, sheet_name="empresa")

#print(df_mayor.head())
#print(df_clients.info())
#print(df_sales.info())
#print(df_enterprise.head())

# fusionar los dos DataFrames utilizando la columna "documents_groups"
merged_df = pd.merge(df_sales, df_clients, left_on='clienteID', right_on='ClienteID', how='left')
# eliminar filas duplicadas basadas en la columna 'documents_groups'
cols_drop= ['Unnamed: 0_x', 'Cliente' , 'Direccion Cliente', 'Unnamed: 0_y','ClienteID']


#  borrar a group of columns
merged_df= merged_df.drop(columns=cols_drop)

#La función lambda comprueba si el valor de 'Fecha Detalle' es nulo, y si es así, lo reemplaza con el valor de la columna 'Fecha Creación'
merged_df['Fecha Detalle'] = merged_df.apply(lambda x: x['Fecha Creacion'] if pd.isnull(x['Fecha Detalle']) else x['Fecha Detalle'], axis=1)

# Supongamos que ahora estamos en septiembre de 2021
merged_df['objective_days'] = round(pd.to_numeric((pd.to_datetime('2021-09-01') - merged_df['Fecha Detalle']) / np.timedelta64(1, 'D')))


#Agregar columnas de calificacion y analisis de los creditos
'''
Condicion de Factura 
-Cancelada: Closed
-NoCancelada: Open


Medición de la Morosidad

https://www.sbs.gob.pe/usuarios/nuestros-servicios/reporte-de-deudas
Las empresas no pueden trabajar sus creditos como los bancos por el tema de la liquidez, en cambio, son un marco de referencia.
Asi como las politicas para otorga creditos a sus clientes

#Calificación	Condición
-Pendiente	    <= - 7       días
-Por Vencer	    <0 ; > -7
-Vencido	    >=0 ; < 7
-Muy Atrasado	>=8  a <= 60 días

Acciones a Tomar:	
#Acciones	                                Intervalo
-Monitorear sin llamar al cliente	        -7
-Llamar al cliente y comunicar deuda	    0
-Visitar al cliente y hacer cobranza	    7
-Negociar y Refinanciar deuda del cliente	60


Nota: No estoy incluyendo una segementacion para cada tipo de cliente.
'''


merged_df['voucher_condition'] = merged_df['DocStatus'].apply(lambda x: 'Cancelada' if x=='Closed' else 'NoCancelada')
merged_df['voucher_condition'] = merged_df['voucher_condition'].astype(str)
print(merged_df['voucher_condition'].dtype)


print(merged_df['objective_days'].dtype)

# change the name of column 'B' to 'C'
merged_df= merged_df.rename(columns={'Razon Social': 'Razon_Social'})
merged_df= merged_df.rename(columns={'Tipo de entrega': 'Tipo_entrega'})



print('###################################################################################1')
def assign_debt_rating(value):
    if isinstance(value, float):
        if value >= 0:
            return '0: Normal'
        elif value >= -8:
            return '1: Problemas potenciales'
        elif value >= -60:
            return '2: Deficiente'
        elif value >= -120:
            return '3: Dudoso'
        else:
            return '4: Pérdida'
    elif isinstance(value, pd.Series):
        if pd.isnull(value['voucher_condition']):
            return 'Unknown'
        elif value['voucher_condition'] == "Cancelada":
            return 'Cerrado'
        elif value['voucher_condition'] == "NoCancelada":
            if value['objective_days'] >= -1 and value['objective_days'] <= 0:
                return 'Pendiente'
            elif value['objective_days'] >= 1 and value['objective_days'] <= 8:
                return '0: Normal'
            elif value['objective_days'] >= 9 and value['objective_days'] <= 30:
                return '1: Problemas potenciales'
            elif value['objective_days'] >= 31 and value['objective_days'] <= 60:
                return '2: Deficiente'
            elif value['objective_days'] >= 61 and value['objective_days'] <= 120:
                return '3: Dudoso'
            elif value['objective_days'] >= 121:
                return '4: Pérdida'
    return ''

merged_df['objective_days'] = merged_df['objective_days'].astype(float)

merged_df['Debt_rating'] = merged_df.apply(assign_debt_rating, axis=1)

print(merged_df['Debt_rating'].unique())


def assign_actions_rating(rows):
    if isinstance(rows, float):
        if rows >= 0:
            return 'Monitorear la deuda'
        elif rows >= -8:
            return 'Llamar al cliente y gestionar la cobranza'
        elif rows >= -120:
            return 'Negociar y Refinanciar deuda del cliente'
        else:
            return 'Evaluar acciones legales y contactar a una agencia'
    elif isinstance(rows, pd.Series):
        if pd.isnull(rows['voucher_condition']):
            return 'Unknown'
        elif rows['voucher_condition'] == "Cancelada":
            return 'Cerrado'
        elif rows['voucher_condition'] == "NoCancelada":
            if rows['objective_days'] >= -1 and rows['objective_days'] <= 0:
                return 'Monitorear la deuda'
            elif rows['objective_days'] >= 1 and rows['objective_days'] <= 8:
                return 'Llamar al cliente y gestionar la cobranza'
            elif rows['objective_days'] >= 9 and rows['objective_days'] <= 30:
                return 'Visitar al cliente y gestionar la cobranza'
            elif rows['objective_days'] >= 31 and rows['objective_days'] <= 120:
                return 'Negociar y Refinanciar deuda del cliente'
            elif rows['objective_days'] >= 121:
                return 'Evaluar acciones legales y contactar a una agencia'
    return ''

merged_df['Acctions_rating'] = merged_df.apply(assign_actions_rating, axis=1)

print(merged_df['Acctions_rating'].unique())

def assign_risk_rating(grade):
    if grade['Debt_rating'] == 'Cerrado':
        return 1
    else:
        return 0


merged_df['risk'] = merged_df.apply(assign_risk_rating, axis=1)

print(merged_df['risk'].unique())

merged_df = merged_df.sort_values("risk", ascending=True)

# Save the dataframe to an Excel file
#merged_df.to_excel('dash.xlsx', index=False)
merged_df.to_excel('data/processed/dashboard.xlsx', index=False)
print('###################################################################################2')


cols_delete= ['ID','Folio','TipoDocumento','Cancelada','DocStatus','Tipo Documento','Corfirmada','TransaccionID','Fecha Creacion','Fecha Detalle','productoID', 'VendedorID','CategoriaID','ProveedorID','Proveedor','clienteID','Nombre del Cliente', 'Telefono', 'Direccion','Moneda_y','email', 'Fecha de Registro', 'Eliminado' ,'Tipo Empresa','Tipo Cliente','objective_days','voucher_condition']
merged_df= merged_df.drop(columns=cols_delete)



# separar los sets de entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(
    merged_df.drop('risk', axis=1),  # predicciones
    merged_df['risk'],  # target
    test_size=0.3,  # porcentaje de observaciones set prueba
    random_state=0)  # semilla para garantizar reproducibilidad

X_train_vr = X_train  #Para comparar las transformaciones de entrenamiento
X_test_vr = X_test #Para comparar las transformaciones de prueba

print(X_train.info())
print(X_train.isnull().mean())



def extract_producto_cat(df, col_names):
    for col_name in col_names:
        # check that column exists and is of string data type
        if col_name not in df.columns or df[col_name].dtype != 'object':
            print(f"Column '{col_name}' not found or not of string data type")
            continue
    
        # extract the first part of the text as a category for the specified column
        df[col_name + '_cat'] = df[col_name].str.extract('([a-zA-Z]+)', expand=False)
        
        # replace NaN values with "Desconocido" for categorical variables
        if df[col_name + '_cat'].dtype == 'object':
            df[col_name + '_cat'].fillna('Desconocido', inplace=True)
        
    return df



cols_prime = ['Producto', 'Moneda_x','Vendedor','Categoria', 'Razon_Social','Ciudad','Pais','Tipo_entrega','Debt_rating', 'Acctions_rating']
X_train_cr = extract_producto_cat(X_train, col_names=cols_prime)
X_test_cr = extract_producto_cat(X_test, col_names=cols_prime)


# drop original columnas
X_train_cr = X_train_cr.drop(columns=cols_prime)
X_test_cr = X_test_cr.drop(columns=cols_prime)

print(X_train_cr.columns)
print(X_test_cr.columns)


print(X_train_cr.info())
print(X_test_cr.info())

print(X_train_cr.isnull().mean())
print(X_test_cr.isnull().mean())
print('###################################################################################3')


# primero vamos a crear una lista, indicando cuales son las 
# variables a sustituir con cada método

features_numeric = ['Cantidad','Precio','Monto','Prioridad','Linea de credito']#,'Producto_num'
features_categoric = ['Producto_cat', 'Moneda_x_cat', 'Vendedor_cat', 'Categoria_cat',
       'Razon_Social_cat', 'Ciudad_cat', 'Pais_cat', 'Tipo_entrega_cat',
       'Debt_rating_cat', 'Acctions_rating_cat']#,'Producto_cat'
#features_categoric = ['Producto','Moneda_x','Vendedor', 'Categoria','Razon_Social','Ciudad','Pais','Tipo_entrega', 'Debt_rating','Acctions_rating']#,'Producto_cat'
# luego ponemos las variables en una lista junto a los transformadores 
# usando el ColumnTransformer

# necesitamos añadir remainder = True para indicar que queremos
# TODAS las columnas devueltas al final de la transformación
# y no solo las que hemos transformado, que es el comportamiento por
# defecto del ColumnTransformer.

preprocessor = ColumnTransformer(transformers=[
    ('numeric_imputer', SimpleImputer(strategy='mean'), features_numeric),
    ('categoric_imputer', SimpleImputer(strategy='most_frequent'), features_categoric)
])


# ahora ajustemos el preprocessor
preprocessor.fit(X_train_cr)

# exploremos los transformers:
preprocessor.transformers

# podemos ver los parámetros aprendidos:

# para el imputer de la media
print(preprocessor.named_transformers_['numeric_imputer'].statistics_)

# y podemos corroborar el valor con el obtenido en el set de entrenamiento
print(X_train_vr[features_numeric].mean())

# para el imputer por la mediana hay dos valores aprendidos
# porque queremos sustituir 2 variables diferentes

print(preprocessor.named_transformers_['categoric_imputer'].statistics_)


# corroboremos estos valores en el segmento de entrenamiento
print(X_train_vr[features_categoric].mode())

# y ahora podemos sustituir los datos
X_train_cr = preprocessor.transform(X_train_cr)
# lo mismo en el set de prueba
X_test_cr = preprocessor.transform(X_test_cr)


print(X_train_vr.shape)

# ahora veamos el resultado de la imputación en el dataframe de 3 columnas
X_train_cr = pd.DataFrame(X_train_cr, columns= features_numeric + features_categoric)
X_test_cr = pd.DataFrame(X_test, columns= features_numeric + features_categoric)

print(X_train_cr.info())
print(X_test_cr.info())


print('###################################################################################4')


SEED = 301
np.random.seed(SEED)

modelo = DummyClassifier()
results = cross_validate(modelo, X_train_cr, y_train, cv = 10, return_train_score=False)
media = results['test_score'].mean()
desviacion_estandar = results['test_score'].std()
print("Accuracy con dummy stratified, 10 = [%.2f, %.2f]" % ((media - 2 * desviacion_estandar)*100, (media + 2 * desviacion_estandar) * 100))


def convert_to_numeric(df, columns):
    """
    Converts specified columns of a dataframe from object to numeric.
    
    Parameters:
        df (pandas.DataFrame): Input dataframe.
        columns (list): List of columns to convert.
    
    Returns:
        pandas.DataFrame: Dataframe with specified columns converted to numeric.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

X_train_cr = convert_to_numeric(X_train_cr, features_numeric)
X_test_cr = convert_to_numeric(X_test_cr, features_numeric)

print(X_train_cr.info())
print(X_test_cr.info())

# check that all categorical variables are present in the dataframe
missing_vars = set(features_categoric) - set(X_train_cr.columns)
if missing_vars:
    raise ValueError(f"Missing categorical variables: {missing_vars}")

# initialize the encoder
ohe_enc = OneHotEncoder(
    top_categories=10,  # adjust as needed
    variables=features_categoric,
    drop_last=False
)

# fit the encoder and transform the data
X_train_cr = ohe_enc.fit_transform(X_train_cr)
X_test_cr = ohe_enc.transform(X_test_cr)

# explore the transformed data
print(X_train_cr.head())




SEED = 301
np.random.seed(SEED)
modelo = DecisionTreeClassifier(max_depth=2)
results = cross_validate(modelo, X_train_cr, y_train, cv = 10, return_train_score=False)
media = results['test_score'].mean()
desviacion_estandar = results['test_score'].std()
print("Accuracy con cross validation, 10 = [%.2f, %.2f]" % ((media - 2 * desviacion_estandar)*100, (media + 2 * desviacion_estandar) * 100))

print('###################################################################################5')

np.random.seed(SEED)
X_train['modelo'] = X_train.Precio + np.random.randint(-2, 3, size=800)
X_train.modelo = X_train.modelo + abs(X_train.modelo.min()) + 1
print(X_train.info())


def iniciar_arbol_de_decision(max_depth):
  SEED = 301
  np.random.seed(SEED)

  cv = GroupKFold(n_splits = 10)
  modelo = DecisionTreeClassifier(max_depth=max_depth)
  results = cross_validate(modelo, X_train_cr, y_train, cv = cv, groups = X_train.modelo, return_train_score=True)
  train_score = results['train_score'].mean()*100
  test_score = results['test_score'].mean()*100
  print('Arbol max_depth = %d, training = %.2f, testing = %.2f' % (max_depth, train_score, test_score))
  tabla = [max_depth, train_score, test_score]
  return tabla

resultados = [iniciar_arbol_de_decision(i) for i in range(1,33)]
resultados = pd.DataFrame(resultados, columns=['max_depth','train','test'])
resultados.head()



sns.lineplot(x='max_depth', y='train', data=resultados)
sns.lineplot(x='max_depth', y='test', data=resultados)
plt.legend(['Train','Test']) 
plt.show()


def iniciar_arbol_de_decision(max_depth, min_samples_leaf):
  SEED = 301
  np.random.seed(SEED)

  cv = GroupKFold(n_splits = 10)
  modelo = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
  results = cross_validate(modelo, X_train_cr, y_train, cv = cv, groups = X_train.modelo, return_train_score=True)
  train_score = results['train_score'].mean()*100
  test_score = results['test_score'].mean()*100
  print('Arbol max_depth = %d, min_samples_leaf = %d, training = %.2f, testing = %.2f' % (max_depth, min_samples_leaf, train_score, test_score))
  tabla = [max_depth, min_samples_leaf, train_score, test_score]
  return tabla

def buscar():
  resultados = []
  for max_depth in range(1,33):
    for min_samples_leaf in [32, 64, 128, 256]:
      tabla = iniciar_arbol_de_decision(max_depth, min_samples_leaf)
      resultados.append(tabla)
  resultados = pd.DataFrame(resultados, columns=['max_depth','min_samples_leaf','train','test'])
  return resultados

resultados = buscar()
resultados.head()

resultados.sort_values('test', ascending = False).head()
print(resultados)
corr = resultados.corr()
print(corr)

sns.heatmap(corr)
plt.show()

from pandas.plotting import scatter_matrix
scatter_matrix(resultados, figsize = (14,8), alpha=0.3)
plt.show()
print('###################################################################################6')

from sklearn.model_selection import KFold

SEED = 301
np.random.seed(SEED)

espacio_de_parametros = {
    'max_depth' : [3,5],
    'min_samples_split' : [32,64,128],
    'min_samples_leaf' : [32,64,128],
    'criterion' : ['gini', 'entropy']
}

buscar = GridSearchCV(DecisionTreeClassifier(),
                      espacio_de_parametros,
                      cv = KFold(n_splits = 5, shuffle=True))

buscar.fit(X_train_cr, y_train)
resultados = pd.DataFrame(buscar.cv_results_)
print(resultados.head())


from sklearn.model_selection import cross_val_score

scores = cross_val_score(buscar, X_train_cr, y_train, cv = KFold(n_splits=5, shuffle=True))
print(scores)

def imprime_score(scores):
  media = scores.mean() * 100
  desviacion = scores.std() * 100
  print("Accuracy media %.2f" % media)
  print("Intervalo [%.2f, %.2f]" % (media - 2 * desviacion, media + 2 * desviacion))

imprime_score(scores) 
mejor = buscar.best_estimator_
print(mejor)


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

features = X_train_cr.columns
plt.figure(figsize=(20,10))
plot_tree(mejor, filled=True, rounded=True, class_names=['no','si'], feature_names=features)
plt.savefig('images/tree.png')
