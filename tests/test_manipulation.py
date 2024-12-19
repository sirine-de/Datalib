import pytest
import pandas as pd
import os
from datalib.manipulation import read_csv, write_csv, fill_missing_values

# Test de la fonction read_csv
def test_read_csv_valid():
    # Créer un DataFrame d'exemple et le sauvegarder dans un fichier CSV temporaire
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    df = pd.DataFrame(data)
    file_path = 'test_file.csv'
    df.to_csv(file_path, index=False)
    
    # Tester la fonction read_csv
    result = read_csv(file_path)
    assert isinstance(result, pd.DataFrame)  # Vérifier que le résultat est un DataFrame
    assert result.shape == (3, 2)  # Vérifier que le DataFrame a 3 lignes et 2 colonnes
    
    # Supprimer le fichier temporaire après test
    os.remove(file_path)

def test_read_csv_invalid():
    # Tester avec un fichier inexistant
    with pytest.raises(ValueError):
        read_csv('non_existent_file.csv')

# Test de la fonction write_csv
def test_write_csv():
    # Créer un DataFrame d'exemple
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    df = pd.DataFrame(data)
    file_path = 'test_output.csv'
    
    # Tester la fonction write_csv
    write_csv(df, file_path)
    
    # Vérifier si le fichier a été créé
    assert os.path.exists(file_path)
    
    # Vérifier si le fichier est un fichier CSV valide
    result = pd.read_csv(file_path)
    assert result.shape == (3, 2)  # Vérifier que le DataFrame a 3 lignes et 2 colonnes
    
    # Supprimer le fichier après test
    os.remove(file_path)

def test_write_csv_invalid():
    # Tester la fonction write_csv avec un chemin invalide
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    df = pd.DataFrame(data)
    invalid_path = '/invalid_path/test_output.csv'
    
    with pytest.raises(ValueError):
        write_csv(df, invalid_path)

# Test de la fonction fill_missing_values
def test_fill_missing_values_mean():
    # Créer un DataFrame avec des valeurs manquantes
    data = {'A': [1, 2, None], 'B': [4, None, 6]}
    df = pd.DataFrame(data)
    
    # Tester la méthode "mean"
    result = fill_missing_values(df, method="mean")
    assert result['A'].isna().sum() == 0  # Vérifier qu'il n'y a plus de valeurs manquantes
    assert result['B'].isna().sum() == 0  # Vérifier qu'il n'y a plus de valeurs manquantes
    assert result['A'][2] == 1.5  # La moyenne de A est 1.5
    assert result['B'][1] == 5.0  # La moyenne de B est 5.0

def test_fill_missing_values_median():
    # Créer un DataFrame avec des valeurs manquantes
    data = {'A': [1, 2, None], 'B': [4, None, 6]}
    df = pd.DataFrame(data)
    
    # Tester la méthode "median"
    result = fill_missing_values(df, method="median")
    assert result['A'].isna().sum() == 0  # Vérifier qu'il n'y a plus de valeurs manquantes
    assert result['B'].isna().sum() == 0  # Vérifier qu'il n'y a plus de valeurs manquantes
    assert result['A'][2] == 1.5  # La médiane de A est 1.5, pas 2
    assert result['B'][1] == 5  # La médiane de B est 5

def test_fill_missing_values_mode():
    # Créer un DataFrame avec des valeurs manquantes
    data = {'A': [1, 2, None], 'B': [4, None, 4]}
    df = pd.DataFrame(data)
    
    # Tester la méthode "mode"
    result = fill_missing_values(df, method="mode")
    assert result['A'].isna().sum() == 0  # Vérifier qu'il n'y a plus de valeurs manquantes
    assert result['B'].isna().sum() == 0  # Vérifier qu'il n'y a plus de valeurs manquantes
    assert result['A'][2] == 1  # Le mode de A est 1
    assert result['B'][1] == 4  # Le mode de B est 4

def test_fill_missing_values_invalid_method():
    # Créer un DataFrame avec des valeurs manquantes
    data = {'A': [1, 2, None], 'B': [4, None, 6]}
    df = pd.DataFrame(data)
    
    # Tester une méthode non valide
    with pytest.raises(ValueError):
        fill_missing_values(df, method="invalid_method")
