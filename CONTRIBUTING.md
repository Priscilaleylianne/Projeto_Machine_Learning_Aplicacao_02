# Guia de Contribuição

## 🚀 Como Começar

### 1. Clone o Repositório
```bash
git clone https://github.com/Priscilaleylianne/Projeto_Machine_Learning_Aplicacao_02.git
cd Projeto_Machine_Learning_Aplicacao_02
```

### 2. Configure o Ambiente
```bash
# Criar ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 3. Prepare os Dados
1. Faça download da base de dados [Cut-Test-Classified Cocoa Beans](https://data.mendeley.com/datasets/jk5h9pfvpz/1)
2. Extraia para uma pasta (ex: `data/cocoa_dataset`)
3. Atualize o caminho no notebook

## 📝 Fluxo de Trabalho Git

### Branches
- `main`: Código estável e testado
- `develop`: Desenvolvimento ativo
- `feature/*`: Novas funcionalidades
- `fix/*`: Correções de bugs

### Fazendo Commits
```bash
# Sempre crie uma branch para suas alterações
git checkout -b feature/sua-feature

# Faça commits frequentes e descritivos
git add .
git commit -m "feat: adiciona extração de features com DenseNet"

# Push para o repositório
git push origin feature/sua-feature
```

### Convenção de Commits
- `feat`: Nova funcionalidade
- `fix`: Correção de bug
- `docs`: Documentação
- `style`: Formatação
- `refactor`: Refatoração de código
- `test`: Testes
- `chore`: Tarefas de manutenção

### Pull Requests
1. Certifique-se de que o código funciona
2. Atualize a documentação se necessário
3. Descreva as mudanças no PR
4. Solicite revisão de um colega

## 📁 Estrutura de Arquivos

```
projeto/
├── notebooks/           # Jupyter notebooks (desenvolvimento)
├── src/                 # Código Python reutilizável
├── docs/                # Documentação e relatório
├── results/             # Resultados dos experimentos
└── README.md            # Documentação principal
```

## 🧪 Executando Experimentos

### No Google Colab
1. Abra o notebook no Colab
2. Monte o Google Drive
3. Configure o caminho da base de dados
4. Execute as células em ordem

### Localmente
```bash
jupyter notebook notebooks/
```

## ✅ Checklist Antes de Commit

- [ ] Código funciona sem erros
- [ ] Notebooks têm outputs limpos ou relevantes
- [ ] Comentários explicativos onde necessário
- [ ] README atualizado (se aplicável)
- [ ] Arquivos grandes NÃO incluídos (modelos .pkl, datasets)

## 🤝 Divisão de Tarefas

| Membro | Responsabilidade |
|--------|------------------|
| Alexandre | Análise exploratória |
| César | Extração de features |
| Ícaro | Treinamento SVM |
| Priscila | Avaliação e relatório |

## 📞 Comunicação

- Issues do GitHub para bugs e features
- PRs para revisão de código
- Commits frequentes para mostrar progresso
