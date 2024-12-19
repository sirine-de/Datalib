import pytest
import numpy as np
import pandas as pd
from datalib.data_statistics import calculate_mean, calculate_median, calculate_mode, calculate_statistics, calculate_std, calculate_correlation

# Test de la fonction calculate_mean
def test_calculate_mean():
    data = [1, 2, 3, 4, 5]
    result = calculate_mean(data)
    assert result == 3  # La moyenne de [1, 2, 3, 4, 5] est 3

def test_calculate_mean_empty():
    data = []
    result = calculate_mean(data)
    assert np.isnan(result)  # La moyenne d'une liste vide doit être NaN

# Test de la fonction calculate_median
def test_calculate_median():
    data = [1, 2, 3, 4, 5]
    result = calculate_median(data)
    assert result == 3  # La médiane de [1, 2, 3, 4, 5] est 3

def test_calculate_median_even():
    data = [1, 2, 3, 4]
    result = calculate_median(data)
    assert result == 2.5  # La médiane de [1, 2, 3, 4] est 2.5

# Test de la fonction calculate_mode
def test_calculate_mode():
    data = [1, 2, 2, 3, 4]
    result = calculate_mode(data)
    assert result == 2  # Le mode de [1, 2, 2, 3, 4] est 2

def test_calculate_mode_multiple_modes():
    data = [1, 1, 2, 2, 3]
    result = calculate_mode(data)
    assert result == 1  # Le premier mode est 1

# Test de la fonction calculate_std
def test_calculate_std():
    data = [1, 2, 3, 4, 5]
    result = calculate_std(data)
    assert result == pytest.approx(1.4142135623730951, rel=1e-9)  # Écart-type de [1, 2, 3, 4, 5]

# Test de la fonction calculate_correlation
def test_calculate_correlation():
    data1 = [1, 2, 3, 4, 5]
    data2 = [5, 4, 3, 2, 1]
    result = calculate_correlation(data1, data2)
    assert result == pytest.approx(-1, abs=1e-9)  # Allow small floating-point error

def test_calculate_correlation_no_relation():
    data1 = [1, 2, 3, 4, 5]
    data2 = [10, 20, 30, 40, 50]
    result = calculate_correlation(data1, data2)
    assert result == 1  # Corrélation parfaite

def test_calculate_correlation_random():
    data1 = np.random.rand(100)
    data2 = np.random.rand(100)
    result = calculate_correlation(data1, data2)
    assert result >= -1 and result <= 1  # La corrélation doit être dans l'intervalle [-1, 1]

# Test de la fonction calculate_statistics
def test_calculate_statistics():
    # Création d'un DataFrame pour tester
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['a', 'b', 'c', 'd', 'e']  # Colonne non numérique pour tester l'exclusion
    })
    
    # Capturer les impressions dans un objet StringIO pour les tester
    from io import StringIO
    import sys
    captured_output = StringIO()
    sys.stdout = captured_output
    
    # Appeler la fonction qui imprime les statistiques
    calculate_statistics(data)
    
    # Vérifier si les statistiques pour les colonnes numériques sont présentes dans la sortie
    output = captured_output.getvalue()
    assert "Mean: 3.0" in output  # Moyenne de A
    assert "Median: 3.0" in output  # Médiane de A
    assert "Mode: 1" in output  # Mode de A
    assert "Standard Deviation: 1.4142135623730951" in output  # Correct Standard Deviation for A
    assert "Mean: 30.0" in output  # Moyenne de B
    assert "Median: 30.0" in output  # Médiane de B
    assert "Mode: 10" in output  # Mode de B
    assert "Standard Deviation: 14.142135623730951" in output  # Correct Standard Deviation for B

    # Restaurer la sortie standard
    sys.stdout = sys.__stdout__