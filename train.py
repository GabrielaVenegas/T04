"""
Script de entrenamiento del modelo
para la predicción del valor de casas
"""

# Librerías necesarias
from datetime import datetime
import logging
import numpy as np
import joblib
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

# Guarda información del día-hora en que corre el código
now = datetime.now()
date_time = now.strftime("%Y%m%d-%H%M%S")
log_train_file_name = f"logs/{date_time}_train.log"

# Setting del log
logging.basicConfig(
    filename=log_train_file_name,
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')
logging.info('Script de entrenamiento')

# Carga de datos limpios
try:
    y_train = np.load('./data/prep/y_train.npy')
    X_train = np.load('./data/prep/X_train.npy')
    logging.info('Carga de data limpia')
except FileNotFoundError:
    logging.error('No se puedo cargar data/prep')
    logging.exception("Ocurrió una excepción", exc_info=True)

# Conteo de registros-columnas
PATH_CLEAN = '.data/prep/'
PATH_MODELO = './artifacts/'
logging.info('Archivo train, columnas: %s', X_train.shape[1])
logging.info('Archivo train, filas: %s', X_train.shape[0])
logging.info('Ruta de input: %s', PATH_CLEAN)
logging.info('Ruta de output: %s', PATH_MODELO)

# Modelo Regresión Lasso
Ls = LassoCV()
Ls.fit(X_train, y_train)
y_hat_prediccion = Ls.predict(X_train)


def rmse(y_true, y_pred):
    """Cálculo de error de la predicción"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


metrica = rmse(y_train, y_hat_prediccion)

# Guardamos modelo
joblib.dump(Ls, 'artifacts/Ls.pkl')

logging.info('El modelo ha sido entrenado con éxito')
logging.info('RMSE para set de entrenamiento con Lasso: %s', metrica)
