# Projeto de Previsão de Preços de Imóveis na Califórnia

Este projeto implementa um modelo de rede neural perceptron (Multi-Layer Perceptron - MLP) usando TensorFlow para prever os preços de imóveis do dataset da Califórnia. O projeto segue uma estrutura modular e implementa as melhores práticas de engenharia de software e ciência de dados.

## Estrutura do Projeto

```
california_housing_predictor/
│
├── data/                 # Dados brutos e processados
│   └── processed/        # Dados após pré-processamento
│
├── notebooks/            # Jupyter notebooks para exploração
│   ├── 1_data_exploration.ipynb
│   ├── 2_model_training_evaluation.ipynb
│   └── 3_results_visualization.ipynb
│
├── src/                  # Código fonte principal modularizado
│   ├── __init__.py
│   ├── data_preprocessing.py  # Funções para carregar e pré-processar dados
│   ├── model.py               # Definição da arquitetura do modelo MLP
│   ├── train.py               # Script para treinar o modelo
│   ├── evaluate.py            # Script para avaliar o modelo
│   ├── predict.py             # Função para fazer previsões
│   └── visualization.py       # Funções para gerar gráficos
│
├── models/               # Modelos treinados salvos
│   └── california_housing_mlp.pkl  # O modelo treinado
│
├── results/              # Resultados gerados (gráficos, tabelas)
│   ├── architecture_performance.csv
│   ├── loss_plot.png
│   ├── accuracy_plot.png
│   ├── prediction_comparison_plot.png
│   └── network_diagram_instructions.txt
│
├── main.py               # Script principal para orquestrar o fluxo
└── README.md             # Esta documentação
```

## Ambiente de Desenvolvimento

* **Linguagem**: Python
* **Bibliotecas Principais**: 
  * TensorFlow/Keras - Para construção e treinamento do modelo de rede neural
  * Pandas - Para manipulação de dados
  * NumPy - Para operações numéricas
  * Scikit-learn - Para carregamento do dataset e métricas de avaliação
  * Matplotlib/Seaborn - Para visualizações
* **Gerenciamento de Dependências**: Poetry (compatível)
* **Versionamento**: Git/GitHub
* **IDE**: VS Code

## Dataset

O projeto utiliza o dataset California Housing, que contém dados sobre preços de casas na Califórnia com base no censo de 1990. O conjunto de dados inclui métricas como:

* Mediana da idade das casas
* Número total de quartos
* Número total de banheiros
* População
* Número de domicílios
* Mediana da renda
* Mediana do valor das casas (variável alvo)
* Entre outras características

## Funcionalidades Principais

### 1. Pré-processamento de Dados (`data_preprocessing.py`)

* Carregamento do dataset California Housing
* Divisão em conjuntos de treino e teste
* Normalização/escalonamento dos dados
* Tratamento de valores ausentes (se necessário)

### 2. Definição do Modelo (`model.py`)

* Implementação de uma arquitetura MLP usando TensorFlow/Keras
* Camadas configuráveis (número de camadas e neurônios)
* Funções de ativação apropriadas para regressão

### 3. Treinamento do Modelo (`train.py`)

* Configuração de otimizador e função de perda
* Implementação do processo de treinamento
* Salvamento do modelo treinado

### 4. Avaliação do Modelo (`evaluate.py`)

* Cálculo de métricas de desempenho (MSE, MAE, R²)
* Avaliação no conjunto de teste
* Geração de relatórios de desempenho

### 5. Predição (`predict.py`)

* Carregamento do modelo treinado
* Interface para fazer previsões com novos dados
* Formatação e transformação inversa das previsões

### 6. Visualização (`visualization.py`)

* Gráficos de evolução do treinamento (loss)
* Comparação entre valores previstos e reais
* Gráficos de métricas de desempenho
* Instruções para gerar um diagrama da rede neural

## Como Utilizar

### Executando o Pipeline Completo

```bash
python main.py --mode full --epochs 100 --batch-size 32 --hidden-layers 64,32
```

### Executando Partes Específicas do Pipeline

1. **Apenas Pré-processamento de Dados**:
```bash
python main.py --mode preprocess
```

2. **Apenas Treinamento**:
```bash
python main.py --mode train --epochs 150 --learning-rate 0.001
```

3. **Apenas Avaliação**:
```bash
python main.py --mode evaluate --model-path models/california_housing_mlp.pkl
```

4. **Apenas Previsões**:
```bash
python main.py --mode predict --model-path models/california_housing_mlp.pkl
```

5. **Apenas Visualizações**:
```bash
python main.py --mode visualize --results-dir custom_results
```

### Parâmetros Configuráveis

* `--mode`: Modo de execução (`full`, `preprocess`, `train`, `evaluate`, `predict`, `visualize`)
* `--model-path`: Caminho para salvar/carregar o modelo
* `--batch-size`: Tamanho do lote para treinamento
* `--epochs`: Número de épocas de treinamento
* `--learning-rate`: Taxa de aprendizado do otimizador
* `--hidden-layers`: Configuração das camadas ocultas (valores separados por vírgula)
* `--results-dir`: Diretório para salvar os resultados
* `--random-seed`: Semente para reprodutibilidade

## Teoria: Redes Neurais Perceptron (MLP)

### O que é um Perceptron?

O Perceptron é a unidade fundamental de uma rede neural artificial, inicialmente proposto por Frank Rosenblatt em 1957. Um perceptron simples recebe várias entradas, aplica pesos a elas, soma os valores ponderados e passa o resultado por uma função de ativação para produzir uma saída.

### Multi-Layer Perceptron (MLP)

Uma MLP consiste em pelo menos três camadas:
1. **Camada de Entrada**: Recebe os dados brutos
2. **Camadas Ocultas**: Realizam transformações nos dados através de pesos e funções de ativação
3. **Camada de Saída**: Produz o resultado final (neste caso, a previsão do preço)

### Backpropagation

O algoritmo de retropropagação (backpropagation) é utilizado para treinar a rede. Ele calcula o gradiente da função de perda em relação a cada peso na rede e usa esses gradientes para atualizar os pesos através de um algoritmo de otimização.

### Funções de Ativação

As funções de ativação introduzem não-linearidade na rede, permitindo que ela aprenda padrões complexos:
- **ReLU (Rectified Linear Unit)**: max(0, x) - Usada nas camadas ocultas
- **Linear**: f(x) = x - Utilizada na camada de saída para problemas de regressão

### Otimizadores

Algoritmos que ajustam os pesos da rede para minimizar a função de perda:
- **Adam**: Combina as vantagens do AdaGrad e RMSProp, adaptando as taxas de aprendizado por parâmetro
- **SGD (Stochastic Gradient Descent)**: Atualiza os pesos usando um subconjunto aleatório de dados

### Funções de Perda para Regressão

- **MSE (Mean Squared Error)**: Média dos quadrados das diferenças entre valores previstos e reais
- **MAE (Mean Absolute Error)**: Média dos valores absolutos das diferenças

### Métricas de Avaliação para Regressão

- **MSE**: Penaliza erros maiores mais fortemente
- **MAE**: Representa o erro médio em unidades da variável alvo
- **R²**: Proporção da variância na variável dependente explicada pelo modelo

## Contribuições e Desenvolvimento Futuro

Possíveis melhorias e extensões para este projeto:

1. **Experimentação com Arquiteturas**: Testar diferentes configurações de camadas e neurônios
2. **Regularização**: Implementar técnicas como dropout e regularização L1/L2
3. **Otimização de Hiperparâmetros**: Implementar busca em grade ou aleatória para encontrar melhores hiperparâmetros
4. **Validação Cruzada**: Implementar k-fold cross-validation para avaliação mais robusta
5. **Feature Engineering**: Criar novas características ou transformações que possam melhorar o desempenho
6. **Comparação de Modelos**: Comparar o desempenho da MLP com outros algoritmos de ML
7. **Deploy**: Implementar uma API REST para servir o modelo em produção

## Autores

[Seu Nome]

## Licença

Este projeto está licenciado sob [sua licença escolhida] - consulte o arquivo LICENSE para detalhes.