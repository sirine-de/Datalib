import pytest
import numpy as np
import pandas as pd
from datalib.visualization import plot_histogram, plot_scatter, plot_correlation_matrix

# Test du tracé d'un histogramme
def test_plot_histogram(monkeypatch):
    # Créer des données de test
    data = np.random.normal(0, 1, 100)
    
    # Utiliser monkeypatch pour éviter l'affichage du graphique lors du test
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    
    # Appeler la fonction de tracé
    plot_histogram(data, title="Test Histogramme", bins=15)
    
    # Le test passe si la fonction s'exécute sans erreur, car plt.show() a été remplacé

# Test du tracé d'un nuage de points
def test_plot_scatter(monkeypatch):
    # Créer des données de test
    x = np.random.rand(100)
    y = np.random.rand(100)
    
    # Utiliser monkeypatch pour éviter l'affichage du graphique lors du test
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    
    # Appeler la fonction de tracé
    plot_scatter(x, y, title="Test Nuage de points")
    
    # Le test passe si la fonction s'exécute sans erreur, car plt.show() a été remplacé

# Test du tracé de la matrice de corrélation
def test_plot_correlation_matrix(monkeypatch):
    # Créer un DataFrame de test
    data = pd.DataFrame({
        "A": np.random.rand(100),
        "B": np.random.rand(100),
        "C": np.random.rand(100)
    })
    
    # Utiliser monkeypatch pour éviter l'affichage du graphique lors du test
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    
    # Appeler la fonction de tracé
    plot_correlation_matrix(data)
    
    # Le test passe si la fonction s'exécute sans erreur, car plt.show() a été remplacé
