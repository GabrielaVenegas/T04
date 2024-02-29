"""
Script de inferencia del modelo
para la predicción del valor de casas
"""

# Librerías necesarias
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import joblib

# Guarda información del día-hora en que corre el código
now = datetime.now()
date_time = now.strftime("%Y%m%d-%H%M%S")
log_inference_file_name = f"logs/{date_time}_inference.log"

# Setting del log
logging.basicConfig(
    filename=log_inference_file_name,
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')
logging.info('Script de inferencia')

# Carga de datos a inferir
try:
    test = np.load('./data/prep/X_test.npy')
    X_test = pd.DataFrame(test)
    logging.info('Carga de data a estimar')
except FileNotFoundError:
    logging.error('No se pudo cargar data a estimar')
    logging.exception("Ocurrió una excepción", exc_info=True)

# Carga de modelo
try:
    Ls = joblib.load('artifacts/Ls.pkl')
    logging.info('Carga de modelo')
except FileNotFoundError:
    logging.error('No se pudo cargar modelo')
    logging.exception("Ocurrió una excepción", exc_info=True)

# Conteo de registros-columnas
PATH_CLEAN = '.data/prep/'
PATH_PRED = '.data/predictions/'
logging.info('Archivo test, columnas: %s', X_test.shape[1])
logging.info('Archivo test, filas: %s', X_test.shape[0])
logging.info('Ruta de input: %s', PATH_CLEAN)
logging.info('Ruta de output: %s', PATH_PRED)

# Obtener predicción
y_prediccion = Ls.predict(X_test)

test_clean = pd.read_csv("./data/raw/test.csv")
test_clean['SalePrice'] = np.exp(y_prediccion)

# Guardar predicción
test_clean.to_csv('data/predictions/test_pred.csv', index=False)
logging.info('Se han generado las predicciones con éxito')
