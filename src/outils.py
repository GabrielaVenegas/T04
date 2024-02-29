"""Módulo con funciones útiles para el repositorio."""

import numpy as np


def guardar(objeto, path):
    """Guarda el objeto en el path solicitado"""
    np.save(path, objeto)
