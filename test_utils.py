"""
Test a las funciones auxiliares
"""
import os
import numpy as np
from src.outils import guardar


def test_guardar(tmpdir):
    """Test"""
    array = np.array([1, 2, 3, 4, 5])
    filename = os.path.join(tmpdir, 'test_array.npy')
    guardar(array, filename)
    expected = np.load(filename)
    assert np.array_equal(array, expected)
