# Repositorio, Inferencia de precio de casa
 
 Este repositorio cuenta con un modelo para la predicción del precio de las casas.
 EL data set de entrenamiento proviene de la siguiente [liga](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) de Kaggle.
 El modelo que se utiliza para la predición es, una regresión lineal simple con regularización Lasso.

 A continuación se presentara la estrucutura del repositorio y las instrucciones para poder utilizarlo.

### Estructura del Repositorio
```bash
.
├── artifacts
│   └── Ls.pkl
├── config.yaml
├── data
│   ├── predictions
│   │   └── test_pred.csv
│   ├── prep
│   │   ├── X_test.npy
│   │   ├── X_train.npy
│   │   └── y_train.npy
│   └── raw
│       ├── data_description.txt
│       ├── test.csv
│       └── train.csv
├── inference.py
├── logs
│   ├── 20240228-204044_prep.log
│   ├── 20240228-204054_train.log
│   └── 20240228-204102_inference.log
├── notebooks
│   └── modelo.py
├── prep.py
├── src
│   ├── __pycache__
│   │   └── outils.cpython-310.pyc
│   └── outils.py
├── test
│   ├── __pycache__
│   │   └── test_utils.cpython-310-pytest-8.0.2.pyc
│   └── test_utils.py
└── train.py
```
### Uso

Para poder usar este repositorio en necesario, clonar el repositorio y correr en el siguiente orden los script.

* prep.py -> pyhton prep.py
* train.py -> python train.py
* inference.py -> python inference.py

#### Resultados "pytest"

1 passed in 0.25s
