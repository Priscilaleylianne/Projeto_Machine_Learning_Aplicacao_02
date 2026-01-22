"""
utils.py

Funções utilitárias para o projeto de classificação de cacau.

Universidade do Estado do Amazonas (UEA)
Machine Learning Aplicado 2
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


# Classes da base de dados
CLASSES = [
    '1_Agglutinated',
    '2_Brittle',
    '3_Compartmentalized_Brown',
    '4_Compartmentalized_Partially_Purple',
    '5_Compartmentalized_Purple',
    '6_Compartmentalized_Slaty',
    '7_Compartmentalized_White',
    '8_Flattened',
    '9_Moldered',
    '10_Plated_Brown',
    '11_Plated_Partially_Purple',
    '12_Plated_Purple',
    '13_Plated_Slaty',
    '14_Plated_White'
]

CLASS_NAMES_SHORT = [
    'Aglutinada', 'Quebradiça', 'Comp.Marrom', 'Comp.P.Violeta',
    'Comp.Violeta', 'Comp.Ardósia', 'Comp.Branca', 'Achatada',
    'Embolorada', 'Chocha.Marrom', 'Chocha.P.Violeta', 'Chocha.Violeta',
    'Chocha.Ardósia', 'Chocha.Branca'
]


def load_dataset(base_path, classes=None):
    """
    Carrega informações da base de dados.
    
    Args:
        base_path: Caminho para a pasta raiz da base
        classes: Lista de classes (opcional, usa padrão)
    
    Returns:
        pd.DataFrame: DataFrame com informações das imagens
    """
    if classes is None:
        classes = CLASSES
    
    data_info = []
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(base_path, class_name)
        
        if not os.path.exists(class_path):
            print(f"⚠️ Diretório não encontrado: {class_path}")
            continue
        
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                data_info.append({
                    'path': os.path.join(class_path, img_name),
                    'class_name': class_name,
                    'class_idx': class_idx,
                    'filename': img_name
                })
    
    df = pd.DataFrame(data_info)
    print(f"✅ Carregadas {len(df)} imagens de {len(df['class_name'].unique())} classes")
    
    return df


def create_holdout_split(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Cria partição holdout estratificada.
    
    Args:
        df: DataFrame com os dados
        test_size: Proporção do conjunto de teste
        val_size: Proporção do conjunto de validação
        random_state: Seed para reprodutibilidade
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df['path'].values
    y = df['class_idx'].values
    
    # Primeiro split: treino+val vs teste
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Segundo split: treino vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio,
        stratify=y_trainval, random_state=random_state
    )
    
    print(f"📊 Partição criada:")
    print(f"   Treino:     {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validação:  {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Teste:      {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_class_distribution(df, save_path=None):
    """
    Plota distribuição de classes.
    """
    class_counts = df['class_name'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = sns.color_palette('husl', len(CLASSES))
    bars = ax.bar(range(len(class_counts)), class_counts.values, color=colors)
    
    ax.set_xticks(range(len(class_counts)))
    ax.set_xticklabels(CLASS_NAMES_SHORT, rotation=45, ha='right')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Quantidade de Imagens')
    ax.set_title('Distribuição de Imagens por Classe')
    
    for bar, count in zip(bars, class_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_sample_images(df, n_samples=1, save_path=None):
    """
    Plota exemplos de imagens de cada classe.
    """
    fig, axes = plt.subplots(2, 7, figsize=(18, 6))
    axes = axes.flatten()
    
    for idx, class_name in enumerate(CLASSES):
        class_df = df[df['class_name'] == class_name]
        if len(class_df) > 0:
            sample = class_df.sample(n_samples).iloc[0]
            img = Image.open(sample['path'])
            axes[idx].imshow(img)
            axes[idx].set_title(CLASS_NAMES_SHORT[idx], fontsize=9)
        axes[idx].axis('off')
    
    plt.suptitle('Exemplos de Imagens por Classe', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title='Matriz de Confusão', save_path=None):
    """
    Plota matriz de confusão.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES_SHORT,
        yticklabels=CLASS_NAMES_SHORT,
        ax=ax
    )
    ax.set_xlabel('Predito', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return cm


def print_metrics(y_true, y_pred):
    """
    Imprime métricas de classificação.
    """
    print("\n" + "="*50)
    print("📊 MÉTRICAS DE AVALIAÇÃO")
    print("="*50)
    print(f"   Acurácia:  {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"   Precisão:  {precision_score(y_true, y_pred, average='weighted')*100:.2f}%")
    print(f"   Revocação: {recall_score(y_true, y_pred, average='weighted')*100:.2f}%")
    print(f"   F1-Score:  {f1_score(y_true, y_pred, average='weighted')*100:.2f}%")


def get_metrics_dict(y_true, y_pred):
    """
    Retorna dicionário com métricas.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }


if __name__ == "__main__":
    print("Módulo de utilidades - Classificação de Cacau")
    print(f"Classes disponíveis: {len(CLASSES)}")
