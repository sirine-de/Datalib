import pytest
import numpy as np
from datalib.analysis import linear_regression, polynomial_regression, kmeans_clustering, pca_analysis, knn_classification, decision_tree_classification
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Test de la régression linéaire
def test_linear_regression():
    # Créer un jeu de données de régression
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
    
    # Appliquer la régression linéaire
    model = linear_regression(X, y)
    
    # Vérifier que le modèle a été correctement ajusté
    assert hasattr(model, 'coef_')  # Vérifier que le modèle a des coefficients
    assert hasattr(model, 'intercept_')  # Vérifier que le modèle a un intercept

# Test de la régression polynomiale
def test_polynomial_regression():
    # Créer un jeu de données de régression
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
    
    # Appliquer la régression polynomiale
    model, poly = polynomial_regression(X, y, degree=3)
    
    # Vérifier que le modèle a été ajusté et la transformation polynomiale effectuée
    assert hasattr(model, 'coef_')  # Vérifier que le modèle a des coefficients
    assert hasattr(model, 'intercept_')  # Vérifier que le modèle a un intercept
    assert poly.degree == 3  # Vérifier que le degré du polynôme est 3

# Test du clustering k-means
def test_kmeans_clustering():
    # Créer un jeu de données de classification
    X, _ = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0)
    
    # Appliquer le clustering k-means
    kmeans = kmeans_clustering(X, n_clusters=3)
    
    # Vérifier que le modèle a été ajusté et a un attribut 'labels_'
    assert hasattr(kmeans, 'labels_')  # Vérifier que les labels des clusters existent
    assert len(np.unique(kmeans.labels_)) == 3  # Vérifier qu'il y a bien 3 clusters

# Test de l'analyse en composantes principales (PCA)
def test_pca_analysis():
    # Créer un jeu de données
    X, _ = make_classification(n_samples=100, n_features=5, n_informative=3)
    
    # Appliquer l'analyse en composantes principales (PCA)
    pca, X_pca = pca_analysis(X, n_components=2)
    
    # Vérifier que le modèle PCA a été ajusté et que les données ont été transformées
    assert hasattr(pca, 'components_')  # Vérifier que le modèle PCA a des composants
    assert X_pca.shape[1] == 2  # Vérifier que les données transformées ont 2 composantes

# Test de la classification k-NN
def test_knn_classification():
    # Créer un jeu de données de classification
    X, y = make_classification(n_samples=100, n_features=3, n_informative=2, n_redundant=0, n_repeated=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Appliquer le classificateur k-NN
    knn = knn_classification(X_train, y_train, n_neighbors=3)
    
    # Vérifier que le modèle a été ajusté et peut être utilisé pour prédire
    assert hasattr(knn, 'predict')  # Vérifier que le modèle a une méthode de prédiction
    predictions = knn.predict(X_test)
    assert predictions.shape[0] == y_test.shape[0]  # Vérifier que la prédiction renvoie un nombre d'échantillons correct

# Test de la classification arbre de décision
def test_decision_tree_classification():
    # Créer un jeu de données de classification
    X, y = make_classification(n_samples=100, n_features=3, n_informative=2, n_redundant=0, n_repeated=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Appliquer le classificateur arbre de décision
    tree = decision_tree_classification(X_train, y_train)
    
    # Vérifier que le modèle a été ajusté et peut être utilisé pour prédire
    assert hasattr(tree, 'predict')  # Vérifier que le modèle a une méthode de prédiction
    predictions = tree.predict(X_test)
    assert predictions.shape[0] == y_test.shape[0]  # Vérifier que la prédiction renvoie un nombre d'échantillons correct