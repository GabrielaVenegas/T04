"""Script inicial para la predicción del valor de casas"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

train = pd.read_csv("../data/raw/train.csv")
# archivo a modificar para predicción
test = pd.read_csv("../data/raw/test.csv")

# vista de los datos de entrenamiento
train.head(2)

# Revisión de variables
data = pd.concat([train.drop('SalePrice', axis=1), test],
                 keys=['train', 'test'])
data.drop(['Id'], axis=1, inplace=True)

years = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
metrics = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
           'BsmtUnfSF',
           'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
           'GarageArea',
           'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
           'ScreenPorch', 'PoolArea',
           'MiscVal']
mask = (data[years] > 2018).any(axis=1)
data.loc[mask, 'GarageYrBlt'] = data[mask]['YearBuilt']

# separación variables numericas y categoricas
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

num = [9, 7, 5, 3, 2]

G = dict(zip(literal, num))

data[grades] = data[grades].replace(G)

cat_feats = data.drop(num_feats, axis=1).columns

# Transformación logarítmica
price = np.log1p(train['SalePrice'])

# asimetría
skewed_feats = data.loc['train'][metrics].apply(lambda x: x.skew(skipna=True))
# compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
data[skewed_feats] = np.log1p(data[skewed_feats])

# Revisión y manejo de datos faltantes

feats = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd',
         'Electrical', 'Functional', 'SaleType']
model = data.loc[
    'train'].groupby('Neighborhood')[feats].apply(lambda
                                                  x: x.mode().iloc[0])

for f in feats:
    data[f].fillna(data['Neighborhood'].map(model[f]), inplace=True)
data['LotFrontage'] = data['LotFrontage'].fillna(data.loc[
    'train', 'LotFrontage'].median())
data['KitchenQual'].fillna(data['OverallQual'], inplace=True)
# reemplazamos con otra variable

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

data['MasVnrType'].replace({'None': np.nan}, inplace=True)

data[cats] = data[cats].fillna('0')
data[nums] = data[nums].fillna(0)

data.isnull().sum().sum()  # revisión final nulos

# cambio de tipo de dato
data['MSSubClass'] = data['MSSubClass'].astype('object', copy=False)
data['MoSold'] = data['MoSold'].astype('object', copy=False)
data['BsmtFullBath'] = data['BsmtFullBath'].astype('int64', copy=False)
data['BsmtHalfBath'] = data['BsmtHalfBath'].astype('int64', copy=False)
data['GarageCars'] = data['GarageCars'].astype('int64', copy=False)
data[years] = data[years].astype('int64', copy=False)


# agrupar variables categoricas
categorical_data = pd.concat((data.loc['train'][cat_feats], price), axis=1)

low = 0.05 * data.loc['train'].shape[0]

for feat in cat_feats:

    order = ((categorical_data.groupby(feat).mean()).
             sort_values(by='SalePrice',
                         ascending=False).index.values.tolist())
    for i in enumerate(order):
        N = (categorical_data[categorical_data[feat] == order[i]]
             .count().max())
        j = i
        while (N < low) & (N != 0):
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
    uniD = data[feat].unique()
    order = categorical_data[feat].unique()

    for i in uniD:
        if i not in order:
            ind = np.argsort(order - i)[0]
            data.replace({feat: {i: order[ind]}}, inplace=True)


# generación dummys
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

X_test = finaldata.loc['test']
X_train = finaldata.loc['train']
y_train = price
m = X_train.mean()
std = X_train.std()

X_train = (X_train - m) / std
X_test = (X_test - m) / std


# Modelo Regresión Lasso
Ls = LassoCV()
Ls.fit(X_train, y_train)
y_prediccion = Ls.predict(X_test)


y_hat_prediccion = Ls.predict(X_train)


def rmse(y_true, y_pred):
    """Cálculo de error de la predicción"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


print("Lasso score on training set: ", rmse(y_train, y_hat_prediccion))

y_prediccion = Ls.predict(X_test)
