# Classificação Inteligente de Sementes de Cacau

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Sobre o Projeto

Trabalho Pedagógico Complementar da disciplina **Machine Learning Aplicado 2** da Universidade do Estado do Amazonas (UEA).

O projeto implementa um sistema de classificação de sementes de cacau no teste de corte utilizando **Transfer Learning** com CNNs pré-treinadas como extratoras de características e **Support Vector Machines (SVM)** como classificadores.

### Objetivo

Classificar automaticamente amêndoas de cacau em 14 classes distintas, seguindo a metodologia de CNNs como extratoras de features + SVMs classificadoras, conforme proposto por [Sarkar et al. (2018)](https://www.packtpub.com/product/hands-on-transfer-learning-with-python/9781788831307).

## Equipe

| Nome | GitHub |
|------|--------|
| Alexandre Teixeira da Silva | [@AlexandreTeixeira](https://github.com/AlexandreTeixeira) |
| César Braz de Oliveira | [@CesarBraz](https://github.com/CesarBraz) |
| Ícaro Guimarães Canto | [@IcaroCanto](https://github.com/IcaroCanto) |
| Priscila Leylianne da Silva Gonçalves | [@Priscilaleylianne](https://github.com/Priscilaleylianne) |

## Base de Dados

Utilizamos a base de dados **Cut-Test-Classified Cocoa Beans** ([Santos et al., 2019](https://www.sciencedirect.com/science/article/pii/S2352340919302331)):

- **Total de imagens:** 1.400
- **Classes:** 14 (100 imagens por classe)
- **Categorias:** Aglutinada, Quebradiça, Compartimentalizada (5 cores), Achatada, Embolorada, Chocha (5 cores)

## Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE CNN + SVM                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌─────────────────┐    ┌──────────────┐    │
│  │ Imagem  │───▶│ CNNs Pré-train. │───▶│   Features   │    │
│  │ 224x224 │    │ (ImageNet)      │    │   Extraídas  │    │
│  └─────────┘    └─────────────────┘    └──────┬───────┘    │
│                                               │             │
│                 ┌─────────────────────────────┘             │
│                 ▼                                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              FEATURE FUSION                           │  │
│  │  InceptionV3 (2048) + VGG16 (512) + EfficientNetB0   │  │
│  │                      (1280)                           │  │
│  │                    = 3840 features                    │  │
│  └──────────────────────────┬───────────────────────────┘  │
│                             │                               │
│                             ▼                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              PCA DIMENSIONALITY REDUCTION             │  │
│  │              (Variância explicada: 99%)               │  │
│  └──────────────────────────┬───────────────────────────┘  │
│                             │                               │
│                             ▼                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    SVM ENSEMBLE                       │  │
│  │    GridSearchCV + Stacking (SVM + LR meta-learner)   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Estrutura do Repositório

```
Projeto_Machine_Learning_Aplicacao_02/
│
├── 📂 notebooks/
│   ├── 01_Analise_Exploratoria.ipynb      # EDA completa da base
│   ├── 02_Extracao_Features.ipynb         # Extração de features CNN
│   ├── 03_Treinamento_SVM.ipynb           # Busca em grade e treino
│   └── 04_Avaliacao_Resultados.ipynb      # Métricas e visualizações
│
├── 📂 src/
│   ├── feature_extraction.py              # Funções de extração
│   ├── svm_classifier.py                  # Pipeline SVM
│   └── utils.py                           # Funções auxiliares
│
├── 📂 docs/
│   └── Relatorio_Final.pdf                # Relatório SBC
│
├── 📂 results/
│   ├── confusion_matrices/                # Matrizes de confusão
│   ├── metrics/                           # Métricas CSV
│   └── models/                            # Modelos salvos
│
├── 📄 README.md
├── 📄 requirements.txt
└── 📄 LICENSE
```

## Metodologia

### 1. Análise Exploratória
- Verificação da distribuição de classes (balanceada: 100/classe)
- Inspeção de qualidade das imagens
- Visualização de exemplos por categoria

### 2. Preparação de Dados
- Redimensionamento para 224×224 pixels
- Normalização dos pixels [0, 1]
- Partição holdout: 70% treino, 10% validação, 20% teste

### 3. Extração de Características
Arquiteturas utilizadas (pesos ImageNet, sem camadas de classificação):
- **InceptionV3** → 2.048 features
- **VGG16** → 512 features  
- **EfficientNetB0** → 1.280 features

### 4. Busca em Grade (SVM)
```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'kernel': ['rbf', 'linear', 'poly']
}
```

### 5. Técnicas de Otimização
- **Feature Fusion:** Concatenação de features múltiplas CNNs
- **PCA:** Redução de dimensionalidade (99% variância)
- **Ensemble Stacking:** SVM + Logistic Regression como meta-learner

## Resultados

### Métricas do Melhor Modelo

| Métrica | Valor |
|---------|-------|
| **Acurácia** | 77.14% ± 2.3% |
| **Precisão** | 77.89% |
| **Revocação** | 77.14% |
| **F1-Score** | 76.82% |

### Comparativo de Abordagens

| Abordagem | Acurácia |
|-----------|----------|
| InceptionV3 + SVM | 72.5% |
| VGG16 + SVM | 68.2% |
| EfficientNetB0 + SVM | 70.1% |
| **Feature Fusion + PCA + Stacking** | **77.14%** |

### Análise Comparativa com a Literatura

| Método | Acurácia | Observação |
|--------|----------|------------|
| Malcher & Guedes (2022) - CNN End-to-End | 89.79% | Fine-tuning completo |
| **Este trabalho - CNN + SVM** | **77.14%** | Transfer learning puro |

> **Nota:** A diferença de ~12% é esperada e documentada na literatura. CNNs com fine-tuning específico para o domínio superam consistentemente abordagens de transfer learning com classificadores tradicionais ([Russakovsky et al., 2015](https://arxiv.org/abs/1409.0575)).

## Como Executar

### Pré-requisitos
```bash
pip install -r requirements.txt
```

### Execução no Google Colab
1. Faça upload da base de dados para o Google Drive
2. Execute os notebooks na ordem numérica
3. Ajuste o caminho da base de dados conforme necessário

### Execução Local
```bash
# Clone o repositório
git clone https://github.com/Priscilaleylianne/Projeto_Machine_Learning_Aplicacao_02.git

# Entre no diretório
cd Projeto_Machine_Learning_Aplicacao_02

# Instale as dependências
pip install -r requirements.txt

# Execute o notebook principal
jupyter notebook notebooks/
```

## Referências

- Santos, F., Palmeira, E., and Jesus, G. (2019). An Image Dataset of Cut-Test-Classified Cocoa Beans. *Data in Brief*, 24:103916.
- Malcher, D. and Guedes, E. (2022). Classificação inteligente do teste de corte do cacau com redes neurais convolucionais profundas. *Anais do XIII WCAMA*, pages 31-40.
- Sarkar, D., Bali, R., and Ghosh, T. (2018). *Hands-On Transfer Learning with Python*. Packt Publishing.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

<p align="center">
  <b>Universidade do Estado do Amazonas (UEA)</b><br>
  Escola Superior de Tecnologia<br>
  Machine Learning Aplicado 2 - Prof. Elloá B. Guedes<br>
  2024/2025
</p>
