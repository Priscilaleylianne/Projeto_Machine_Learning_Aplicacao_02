"""
feature_extraction.py

Módulo para extração de características de imagens usando CNNs pré-treinadas.

Universidade do Estado do Amazonas (UEA)
Machine Learning Aplicado 2
"""

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import (
    VGG16, InceptionV3, EfficientNetB0,
    MobileNetV2, ResNet50, DenseNet121
)
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing import image


# Configurações das arquiteturas suportadas
ARCHITECTURES = {
    'VGG16': {
        'model_fn': VGG16,
        'preprocess': vgg_preprocess,
        'input_size': (224, 224),
        'output_dim': 512
    },
    'InceptionV3': {
        'model_fn': InceptionV3,
        'preprocess': inception_preprocess,
        'input_size': (299, 299),
        'output_dim': 2048
    },
    'EfficientNetB0': {
        'model_fn': EfficientNetB0,
        'preprocess': efficientnet_preprocess,
        'input_size': (224, 224),
        'output_dim': 1280
    }
}


def create_feature_extractor(model_name):
    """
    Cria um modelo extrator de características.
    
    Args:
        model_name (str): Nome da arquitetura ('VGG16', 'InceptionV3', 'EfficientNetB0')
    
    Returns:
        tuple: (model, preprocess_fn, input_size)
    """
    if model_name not in ARCHITECTURES:
        raise ValueError(f"Arquitetura '{model_name}' não suportada. "
                        f"Opções: {list(ARCHITECTURES.keys())}")
    
    config = ARCHITECTURES[model_name]
    
    # Criar modelo sem camadas de classificação
    base_model = config['model_fn'](
        weights='imagenet',
        include_top=False,
        pooling='avg'
    )
    
    # Congelar pesos
    base_model.trainable = False
    
    return base_model, config['preprocess'], config['input_size']


def extract_features(image_paths, model, preprocess_fn, input_size, batch_size=32):
    """
    Extrai características de um conjunto de imagens.
    
    Args:
        image_paths: Lista de caminhos das imagens
        model: Modelo Keras extrator
        preprocess_fn: Função de pré-processamento
        input_size: Tupla (height, width)
        batch_size: Tamanho do batch
    
    Returns:
        numpy.ndarray: Array de características (n_samples, n_features)
    """
    features_list = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extraindo features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            # Carregar e redimensionar imagem
            img = image.load_img(img_path, target_size=input_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_fn(img_array)
            batch_images.append(img_array)
        
        # Processar batch
        batch_array = np.vstack(batch_images)
        batch_features = model.predict(batch_array, verbose=0)
        features_list.append(batch_features)
    
    return np.vstack(features_list)


def extract_features_multi_cnn(image_paths, architectures=None, batch_size=32):
    """
    Extrai e concatena características de múltiplas CNNs.
    
    Args:
        image_paths: Lista de caminhos das imagens
        architectures: Lista de arquiteturas (default: todas)
        batch_size: Tamanho do batch
    
    Returns:
        numpy.ndarray: Features concatenadas
    """
    if architectures is None:
        architectures = list(ARCHITECTURES.keys())
    
    all_features = []
    
    for arch in architectures:
        print(f"\n[{arch}] Extraindo características...")
        
        model, preprocess_fn, input_size = create_feature_extractor(arch)
        features = extract_features(
            image_paths, model, preprocess_fn, input_size, batch_size
        )
        all_features.append(features)
        
        # Liberar memória
        del model
        tf.keras.backend.clear_session()
    
    # Concatenar features (Feature Fusion)
    fused_features = np.hstack(all_features)
    print(f"\n✅ Feature Fusion: {fused_features.shape[1]} features totais")
    
    return fused_features


if __name__ == "__main__":
    # Teste básico
    print("Arquiteturas disponíveis:")
    for name, config in ARCHITECTURES.items():
        print(f"  - {name}: {config['output_dim']} features, input {config['input_size']}")
