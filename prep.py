"""
Script de preprocesamiento para la predicción del valor de casas
"""
# Librerías necesarias
from datetime import datetime
import logging
import numpy as np
import pandas as pd
from src.outils import guardar

# Guarda información del día-hora en que corre el código
now = datetime.now()
date_time = now.strftime("%Y%m%d-%H%M%S")
log_prep_file_name = f"logs/{date_time}_prep.log"

# Setting del log
logging.basicConfig(
    filename=log_prep_file_name,
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

logging.info('Script de preprocesamiento')

# Este archivo sirve para entrenar el modelo
PATH_RAW = './data/raw/'
PATH_CLEAN = '.data/prep/'
try:
    train = pd.read_csv("./data/raw/train.csv")
    test = pd.read_csv("./data/raw/test.csv")
    logging.info('Carga de datos de entrenamiento y test')
except FileNotFoundError:
    logging.error('No se puedo cargar archivo train o test')
    logging.exception("Ocurrió una excepción", exc_info=True)

# Conteo de registros-columnas
logging.info('Archivo train, columnas: %s', train.shape[1])
logging.info('Archivo train, filas: %s', train.shape[0])
logging.info('Archivo test, columnas: %s', test.shape[1])
logging.info('Archivo test, filas: %s', test.shape[0])
logging.info('Ruta de input: %s', PATH_RAW)
logging.info('Ruta de output: %s', PATH_CLEAN)

# Concatena train y test, elimina variable precio
data = pd.concat([train.drop('SalePrice', axis=1), test],
                 keys=['train', 'test'])
data.drop(['Id'], axis=1, inplace=True)

# Agrupamos las variables que contienen información en años
years = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']

# Agrupamos las variables de medida
metrics = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
           'BsmtUnfSF',
           'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
           'GarageArea',
           'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
           'ScreenPorch', 'PoolArea',
           'MiscVal']

# Quitamos atípicos de año en GarageYrBlt y asignamos el año de YearBuilt
# Poner este rev, para generalizar falta
mask = (data[years] > 2018).any(axis=1)
data.loc[mask, 'GarageYrBlt'] = data[mask]['YearBuilt']

# Separación variables numericas y categoricas
num_feats = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
             'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'ExterQual',
             'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtFinSF1',
             'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC',
             '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
             'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
             'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
             'Fireplaces', 'FireplaceQu', 'GarageYrBlt',
             'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
             'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
             'ScreenPorch', 'PoolArea', 'PoolQC', 'MiscVal',
             'YrSold']

grades = ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond',
          'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual',
          'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

literal = ['Ex', 'Gd', 'TA', 'Fa', 'Po']

# Escala nueva para variables categóricas
num = [9, 7, 5, 3, 2]
G = dict(zip(literal, num))
data[grades] = data[grades].replace(G)

cat_feats = data.drop(num_feats, axis=1).columns

# Transformación logarítmica de la variable a predecir
price = np.log1p(train['SalePrice'])

# Revisión asimetría de variables de medida
skewed_feats = data.loc['train'][metrics].apply(lambda x: x.skew(skipna=True))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

# Transformación logarítmica de variables de medida
data[skewed_feats] = np.log1p(data[skewed_feats])

# falta agregar revisión data missing, mejorar, muy manual
# Revisión y manejo de datos faltantes
feats = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd',
         'Electrical', 'Functional', 'SaleType']
# Llenar los faltantes más importantes
# Moda
model = data.loc[
    'train'].groupby('Neighborhood')[feats].apply(lambda
                                                  x: x.mode().iloc[0])
# Mediana
for f in feats:
    data[f].fillna(data['Neighborhood'].map(model[f]), inplace=True)
data['LotFrontage'] = data['LotFrontage'].fillna(data.loc[
    'train', 'LotFrontage'].median())
# Reemplazamos con otra variable
data['KitchenQual'].fillna(data['OverallQual'], inplace=True)

# Variables sin valor por NA
bsmt = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',
        'BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF']
fire = ['Fireplaces', 'FireplaceQu']
garage = ['GarageQual', 'GarageCond', 'GarageType', 'GarageFinish',
          'GarageCars', 'GarageArea', 'GarageYrBlt']
masn = ['MasVnrType', 'MasVnrArea']
others = ['Alley', 'Fence', 'PoolQC', 'MiscFeature']
cats = data.columns[data.dtypes == 'object']
nums = list(set(data.columns) - set(cats))
data['MasVnrType'].replace({'None': np.nan}, inplace=True)  # NA y None

# Los mandamos a cero
data[cats] = data[cats].fillna('0')
data[nums] = data[nums].fillna(0)

# Revisión final nulos
# Falta ponerlo aparte
data.isnull().sum().sum()

# Cambio de tipo de dato
data['MSSubClass'] = data['MSSubClass'].astype('object', copy=False)
data['MoSold'] = data['MoSold'].astype('object', copy=False)
data['BsmtFullBath'] = data['BsmtFullBath'].astype('int64', copy=False)
data['BsmtHalfBath'] = data['BsmtHalfBath'].astype('int64', copy=False)
data['GarageCars'] = data['GarageCars'].astype('int64', copy=False)
data[years] = data[years].astype('int64', copy=False)

# Agrupar variables categoricas con etiquetas de poca frecuencia
categorical_data = pd.concat((data.loc['train'][cat_feats], price), axis=1)
# definimos densidad de etiqueta de al menos 5%
# falta ponerlo aparte param
inferior = 0.05 * data.loc['train'].shape[0]
for feat in cat_feats:
    order = ((categorical_data.groupby(feat).mean()).
             sort_values(by='SalePrice',   # Agrupamos respecto a precio
                         ascending=False).index.values.tolist())
    ii = len(order)
    for i in range(0, ii):
        N = (categorical_data[categorical_data[feat] == order[i]]
             .count().max())
        j = i
        while (N < inferior) & (N != 0):
            j += 1
            if j > len(order) - 1:
                j = i - 1
                break
            N += (categorical_data[categorical_data[feat] == order[j]]
                  .count().max())
        if j < i:
            lim = len(order)
        else:
            lim = j
        for k in range(i, lim):
            categorical_data.replace({feat: {order[k]: order[j]}},
                                     inplace=True)
            data.replace({feat: {order[k]: order[j]}},
                         inplace=True)
    unicos = data[feat].unique()  # Valores únicos
    order = categorical_data[feat].unique()
    for i in unicos:
        if i not in order:
            ind = np.argsort(order - i)[0]
            data.replace({feat: {i: order[ind]}}, inplace=True)


# Generación dummys
for feat in categorical_data.columns[:-1]:
    uni = categorical_data.groupby(
        feat).mean().sort_values(by='SalePrice').index
    if len(uni) < 2:
        data.drop(feat, axis=1, inplace=True)
    elif len(uni) < 3:
        data[feat].replace({uni[0]: 0, uni[1]: 1}, inplace=True)
        data[feat] = data[feat].astype('int8')
    else:
        data[feat] = data[feat].astype('category')
finaldata = pd.get_dummies(data)
black_list = bsmt + fire + garage + masn + others
for feat in finaldata.columns:
    if ('_0' in feat) and (feat.split("_")[0] in black_list):
        finaldata.drop(feat, axis=1, inplace=True)

# Reescalar para una mejor regresión
X_test = finaldata.loc['test']
X_train = finaldata.loc['train']
y_train = price

m = X_train.mean()
std = X_train.std()
X_train = (X_train - m) / std
X_test = (X_test - m) / std

logging.info('Los datos han sido preprocesados con éxito')

# Guardar datos procesados
guardar(y_train, 'data/prep/y_train.npy')
guardar(X_train, 'data/prep/X_train.npy')
guardar(X_test, 'data/prep/X_test.npy')
