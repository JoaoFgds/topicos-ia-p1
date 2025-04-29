"""
main.py

Script principal que orquestra todo o fluxo de trabalho do projeto de
previsão de preços de imóveis na Califórnia, desde o carregamento e
pré-processamento dos dados até a avaliação e visualização dos resultados.

Este script serve como ponto de entrada principal para executar o pipeline
completo de machine learning e demonstra a integração entre os diferentes
módulos do projeto.
"""

import os
import sys
import time
import argparse
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# Importar módulos do projeto
from src.data_preprocessing import load_and_preprocess_data
from src.model import create_mlp_model
from src.train import train_model, save_model
from src.evaluate import load_model, evaluate_model, calculate_metrics
from src.predict import predict_housing_prices
from src.visualization import (
    plot_training_history,
    plot_predictions_comparison,
    generate_architecture_performance_table,
    create_network_diagram_instructions,
    plot_feature_importance,
)


def parse_arguments():
    """
    Analisa os argumentos de linha de comando para o script.

    Returns:
        Os argumentos analisados.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline de previsão de preços de imóveis da Califórnia"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "preprocess", "train", "evaluate", "predict", "visualize"],
        help="Modo de execução do pipeline",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/california_housing_mlp.pkl",
        help="Caminho para salvar/carregar o modelo treinado",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamanho do lote (batch size) para treinamento",
    )

    parser.add_argument(
        "--epochs", type=int, default=100, help="Número de épocas para treinamento"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Taxa de aprendizado do otimizador",
    )

    parser.add_argument(
        "--hidden-layers",
        type=str,
        default="64,32",
        help="Configuração de camadas ocultas (valores separados por vírgula)",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Diretório para salvar os resultados",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Semente para geradores de números aleatórios (para reprodutibilidade)",
    )

    return parser.parse_args()


def setup_directories(directories):
    """
    Cria os diretórios necessários para o projeto.

    Args:
        directories: Lista de diretórios a serem criados.
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Diretório verificado/criado: {directory}")


def run_pipeline(args):
    """
    Executa o pipeline completo ou partes dele com base nos argumentos.

    Args:
        args: Argumentos de linha de comando analisados.
    """
    # Configurar diretórios
    setup_directories(["data", "models", args.results_dir])

    # Configurar seed para reprodutibilidade
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    # Configurar camadas ocultas
    hidden_layers = [int(units) for units in args.hidden_layers.split(",")]

    # Executar com base no modo selecionado
    if args.mode in ["full", "preprocess"]:
        print("\n--- Carregando e pré-processando dados ---")
        X_train, X_test, y_train, y_test, scaler_X, scaler_y = (
            load_and_preprocess_data()
        )
        print(
            f"Dados carregados e pré-processados: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"
        )

        # Salvar os scalers para uso futuro
        joblib.dump(scaler_X, "models/scaler_X.pkl")
        joblib.dump(scaler_y, "models/scaler_y.pkl")

    if args.mode == "preprocess":
        # Se o modo for apenas pré-processar, paramos aqui
        return

    if args.mode in ["full", "train"]:
        # Se não estivermos continuando de um 'preprocess', precisamos carregar os dados
        if args.mode != "full":
            print("\n--- Carregando e pré-processando dados ---")
            X_train, X_test, y_train, y_test, scaler_X, scaler_y = (
                load_and_preprocess_data()
            )
            print(
                f"Dados carregados e pré-processados: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"
            )

        print("\n--- Criando e treinando o modelo ---")
        # Obter o número de features dos dados
        n_features = X_train.shape[1]

        # Criar o modelo MLP
        model = create_mlp_model(
            input_dim=n_features,
            hidden_layers=hidden_layers,
            learning_rate=args.learning_rate,
        )

        # Treinar o modelo e medir o tempo de treinamento
        start_time = time.time()
        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,  # Usando conjunto de teste como validação
            y_val=y_test,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )
        training_time = time.time() - start_time
        print(f"Tempo de treinamento: {training_time:.2f} segundos")

        # Salvar o modelo
        save_model(model, args.model_path)
        print(f"Modelo salvo em {args.model_path}")

        # Salvar o histórico de treinamento
        joblib.dump(history.history, f"{args.results_dir}/training_history.pkl")

    if args.mode in ["full", "evaluate", "predict", "visualize"]:
        # Se não estivermos continuando de um modo anterior, precisamos carregar dados e modelo
        if args.mode not in ["full", "train"]:
            print("\n--- Carregando dados e modelo ---")
            X_train, X_test, y_train, y_test, scaler_X, scaler_y = (
                load_and_preprocess_data()
            )

            try:
                model = load_model(args.model_path)
                history = joblib.load(f"{args.results_dir}/training_history.pkl")
                training_time = (
                    0  # Não temos essa informação se apenas carregamos o modelo
                )
            except FileNotFoundError:
                print(
                    f"ERRO: Modelo não encontrado em {args.model_path} ou histórico não encontrado."
                )
                print(
                    "Execute o modo 'train' primeiro ou especifique um caminho válido."
                )
                return

    if args.mode in ["full", "evaluate", "predict"]:
        print("\n--- Avaliando o modelo no conjunto de teste ---")
        test_loss, test_metrics = evaluate_model(model, X_test, y_test)
        print(f"Perda no teste: {test_loss:.4f}")
        for metric_name, metric_value in test_metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

    if args.mode in ["full", "predict"]:
        print("\n--- Gerando previsões ---")
        y_pred = predict_housing_prices(model, X_test)

        # Se os targets foram escalonados, desescalonar as previsões e valores reais para exibição
        if scaler_y is not None:
            y_test_original = scaler_y.inverse_transform(
                y_test.reshape(-1, 1)
            ).flatten()
            y_pred_original = scaler_y.inverse_transform(
                y_pred.reshape(-1, 1)
            ).flatten()
        else:
            y_test_original = y_test
            y_pred_original = y_pred

        # Mostrar algumas previsões
        print("\nAmostras de previsões:")
        for i in range(min(5, len(y_pred_original))):
            print(
                f"Real: ${y_test_original[i]*1000:.2f}, "
                f"Previsto: ${y_pred_original[i]*1000:.2f}, "
                f"Erro: ${(y_test_original[i] - y_pred_original[i])*1000:.2f}"
            )

        # Salvar previsões em CSV
        results_df = pd.DataFrame(
            {
                "Valor Real": y_test_original,
                "Valor Previsto": y_pred_original,
                "Erro": y_test_original - y_pred_original,
            }
        )
        results_df.to_csv(f"{args.results_dir}/predictions.csv", index=False)
        print(f"Previsões salvas em {args.results_dir}/predictions.csv")

    if args.mode in ["full", "visualize"]:
        print("\n--- Gerando visualizações ---")

        # Plotar histórico de treinamento
        plot_training_history(history.history, args.results_dir)
        print(f"Gráficos de histórico de treinamento salvos em {args.results_dir}")

        # Plotar comparação entre valores reais e previstos
        if "y_pred" not in locals():
            y_pred = model.predict(X_test).flatten()

        plot_predictions_comparison(
            y_test, y_pred, f"{args.results_dir}/prediction_comparison_plot.png"
        )
        print(
            f"Gráfico de comparação de previsões salvo em {args.results_dir}/prediction_comparison_plot.png"
        )

        # Gerar tabela de desempenho da arquitetura
        generate_architecture_performance_table(
            model=model,
            history=history.history,
            training_time=training_time if "training_time" in locals() else 0,
            test_loss=test_loss if "test_loss" in locals() else 0,
            test_metrics=test_metrics if "test_metrics" in locals() else {},
            output_path=f"{args.results_dir}/architecture_performance.csv",
        )
        print(
            f"Tabela de desempenho da arquitetura salva em {args.results_dir}/architecture_performance.csv"
        )

        # Gerar instruções para diagrama da rede
        create_network_diagram_instructions(
            model, f"{args.results_dir}/network_diagram_instructions.txt"
        )
        print(
            f"Instruções para o diagrama da rede salvas em {args.results_dir}/network_diagram_instructions.txt"
        )

        # Tentar gerar gráfico de importância das features (se aplicável)
        try:
            # Obter nomes das features do dataset (na ausência de nomes reais, usamos índices)
            feature_names = [f"Feature_{i+1}" for i in range(X_train.shape[1])]

            plot_feature_importance(
                model=model,
                feature_names=feature_names,
                output_path=f"{args.results_dir}/feature_importance.png",
            )
            print(
                f"Gráfico de importância das features salvo em {args.results_dir}/feature_importance.png"
            )
        except Exception as e:
            print(f"Não foi possível gerar o gráfico de importância das features: {e}")

    print("\n--- Pipeline concluído com sucesso! ---")


if __name__ == "__main__":
    # Configurar a repetibilidade para o TensorFlow
    tf.keras.utils.set_random_seed(42)

    # Analisar argumentos da linha de comando
    args = parse_arguments()

    # Executar o pipeline
    run_pipeline(args)
