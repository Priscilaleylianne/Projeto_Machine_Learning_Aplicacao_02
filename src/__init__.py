"""
Módulo de classificação de sementes de cacau.

Universidade do Estado do Amazonas (UEA)
Machine Learning Aplicado 2
"""

from .feature_extraction import (
    create_feature_extractor,
    extract_features,
    extract_features_multi_cnn,
    ARCHITECTURES
)

from .svm_classifier import (
    CocoaSVMClassifier,
    StackingSVMClassifier,
    DEFAULT_PARAM_GRID
)

from .utils import (
    load_dataset,
    create_holdout_split,
    plot_class_distribution,
    plot_sample_images,
    plot_confusion_matrix,
    print_metrics,
    get_metrics_dict,
    CLASSES,
    CLASS_NAMES_SHORT
)

__all__ = [
    # Feature extraction
    'create_feature_extractor',
    'extract_features',
    'extract_features_multi_cnn',
    'ARCHITECTURES',
    
    # Classifiers
    'CocoaSVMClassifier',
    'StackingSVMClassifier',
    'DEFAULT_PARAM_GRID',
    
    # Utils
    'load_dataset',
    'create_holdout_split',
    'plot_class_distribution',
    'plot_sample_images',
    'plot_confusion_matrix',
    'print_metrics',
    'get_metrics_dict',
    'CLASSES',
    'CLASS_NAMES_SHORT'
]

__version__ = '1.0.0'
