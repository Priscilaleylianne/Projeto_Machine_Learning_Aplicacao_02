"""
svm_classifier.py

Módulo para classificação com SVM e busca em grade.

Universidade do Estado do Amazonas (UEA)
Machine Learning Aplicado 2
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
import time


# Grade de hiperparâmetros padrão
DEFAULT_PARAM_GRID = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear', 'poly']
}


class CocoaSVMClassifier:
    """
    Classificador SVM otimizado para classificação de cacau.
    """
    
    def __init__(self, pca_variance=0.99, random_state=42):
        """
        Inicializa o classificador.
        
        Args:
            pca_variance: Variância a ser mantida pelo PCA
            random_state: Seed para reprodutibilidade
        """
        self.pca_variance = pca_variance
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.best_model = None
        self.grid_search_results = None
    
    def fit(self, X_train, y_train, param_grid=None, cv=5):
        """
        Treina o classificador com busca em grade.
        
        Args:
            X_train: Features de treino
            y_train: Labels de treino
            param_grid: Grade de hiperparâmetros (opcional)
            cv: Número de folds para validação cruzada
        
        Returns:
            self
        """
        if param_grid is None:
            param_grid = DEFAULT_PARAM_GRID
        
        # Normalizar
        print("📊 Normalizando features...")
        X_scaled = self.scaler.fit_transform(X_train)
        
        # PCA
        print(f"📉 Aplicando PCA ({self.pca_variance*100:.0f}% variância)...")
        self.pca = PCA(n_components=self.pca_variance)
        X_pca = self.pca.fit_transform(X_scaled)
        print(f"   Dimensões: {X_train.shape[1]} → {X_pca.shape[1]}")
        
        # Busca em grade
        print("\n🔍 Iniciando busca em grade...")
        svm = SVC(random_state=self.random_state, probability=True)
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            estimator=svm,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        start_time = time.time()
        grid_search.fit(X_pca, y_train)
        elapsed = time.time() - start_time
        
        self.best_model = grid_search.best_estimator_
        self.grid_search_results = pd.DataFrame(grid_search.cv_results_)
        
        print(f"\n✅ Busca concluída em {elapsed/60:.2f} min")
        print(f"🏆 Melhores parâmetros: {grid_search.best_params_}")
        print(f"📈 Melhor acurácia CV: {grid_search.best_score_*100:.2f}%")
        
        return self
    
    def predict(self, X):
        """
        Realiza predição.
        
        Args:
            X: Features para predição
        
        Returns:
            numpy.ndarray: Predições
        """
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.best_model.predict(X_pca)
    
    def predict_proba(self, X):
        """
        Retorna probabilidades de predição.
        """
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.best_model.predict_proba(X_pca)
    
    def evaluate(self, X_test, y_test, class_names=None):
        """
        Avalia o modelo no conjunto de teste.
        
        Args:
            X_test: Features de teste
            y_test: Labels de teste
            class_names: Nomes das classes (opcional)
        
        Returns:
            dict: Métricas de avaliação
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print("\n" + "="*50)
        print("📊 MÉTRICAS DE AVALIAÇÃO")
        print("="*50)
        print(f"   Acurácia:  {metrics['accuracy']*100:.2f}%")
        print(f"   Precisão:  {metrics['precision']*100:.2f}%")
        print(f"   Revocação: {metrics['recall']*100:.2f}%")
        print(f"   F1-Score:  {metrics['f1_score']*100:.2f}%")
        
        if class_names:
            print("\n📋 Relatório de Classificação:")
            print(classification_report(y_test, y_pred, target_names=class_names))
        
        return metrics
    
    def save(self, filepath):
        """Salva o modelo treinado."""
        joblib.dump(self, filepath)
        print(f"✅ Modelo salvo em {filepath}")
    
    @staticmethod
    def load(filepath):
        """Carrega um modelo salvo."""
        return joblib.load(filepath)


class StackingSVMClassifier(CocoaSVMClassifier):
    """
    Classificador com Stacking de múltiplos SVMs.
    """
    
    def fit(self, X_train, y_train, cv=5):
        """
        Treina o ensemble com stacking.
        """
        # Normalizar
        print("📊 Normalizando features...")
        X_scaled = self.scaler.fit_transform(X_train)
        
        # PCA
        print(f"📉 Aplicando PCA ({self.pca_variance*100:.0f}% variância)...")
        self.pca = PCA(n_components=self.pca_variance)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Criar estimadores base
        print("\n🔧 Criando ensemble com Stacking...")
        estimators = [
            ('svm_rbf', SVC(kernel='rbf', C=10, gamma='scale', 
                          probability=True, random_state=self.random_state)),
            ('svm_linear', SVC(kernel='linear', C=1, 
                              probability=True, random_state=self.random_state)),
            ('svm_poly', SVC(kernel='poly', C=10, gamma='scale',
                           probability=True, random_state=self.random_state))
        ]
        
        meta_learner = LogisticRegression(max_iter=1000, random_state=self.random_state)
        
        self.best_model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=cv,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        print("🚀 Treinando ensemble...")
        start_time = time.time()
        self.best_model.fit(X_pca, y_train)
        
        print(f"✅ Treinamento concluído em {time.time()-start_time:.2f}s")
        
        return self


if __name__ == "__main__":
    # Teste básico
    print("Módulo de classificação SVM para cacau")
    print(f"Grade padrão: {DEFAULT_PARAM_GRID}")
