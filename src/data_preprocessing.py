"""
Módulo para carregamento e pré-processamento do dataset California Housing.

Este módulo contém funções para carregar o dataset, dividir em conjuntos de treino
e teste, e aplicar as transformações necessárias como normalização/padronização.
"""

import os
import joblib
import numpy as np
import pandas as pd

from typing import Tuple, Dict, Any

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_california_housing_data() -> pd.DataFrame:
    """
    Carrega o dataset California Housing da biblioteca scikit-learn.

    Returns:
        pd.DataFrame: DataFrame contendo os dados do California Housing.
    """
    # Carregar o dataset
    housing = fetch_california_housing()

    # Criar DataFrame com os dados
    df = pd.DataFrame(data=housing.data, columns=housing.feature_names)

    # Adicionar a coluna target (preço médio das casas)
    df["MedHouseVal"] = housing.target

    print(f"Dataset carregado com sucesso. Shape: {df.shape}")
    print(f"Features: {housing.feature_names}")
    print(f"Target: MedHouseVal (Valor médio das casas)")

    return df


def check_missing_values(df: pd.DataFrame) -> None:
    """
    Verifica e exibe informações sobre valores ausentes no DataFrame.

    Args:
        df (pd.DataFrame): DataFrame a ser verificado.
    """
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    missing_info = pd.DataFrame(
        {"Valores Ausentes": missing_values, "Percentual (%)": missing_percentage}
    )

    print("\nVerificação de valores ausentes:\n")
    print(missing_info[missing_info["Valores Ausentes"] > 0])

    if missing_info["Valores Ausentes"].sum() == 0:
        print("O dataset não possui valores ausentes!")


def preprocess_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Realiza o pré-processamento dos dados, incluindo a divisão em conjuntos de treino e teste,
    e a normalização das features.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        test_size (float): Proporção dos dados a serem usados como teste (entre 0 e 1).
        random_state (int): Seed para garantir reprodutibilidade.

    Returns:
        Tuple contendo:
            X_train (np.ndarray): Features do conjunto de treino.
            X_test (np.ndarray): Features do conjunto de teste.
            y_train (np.ndarray): Target do conjunto de treino.
            y_test (np.ndarray): Target do conjunto de teste.
            preprocessors (Dict): Dicionário contendo os objetos de pré-processamento (ex: scaler).
    """
    # Verificar valores ausentes
    check_missing_values(df)

    # Separar features e target
    X = df.drop("MedHouseVal", axis=1).values
    y = df["MedHouseVal"].values

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(
        f"\nDivisão em treino/teste:\n\n{X_train.shape[0]} amostras de treino, {X_test.shape[0]} amostras de teste."
    )

    # Normalizar as features usando StandardScaler
    # Importante: o scaler deve ser ajustado apenas nos dados de treino para evitar data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Dicionário para armazenar os objetos de pré-processamento
    preprocessors = {"scaler": scaler}

    print("Normalização aplicada com StandardScaler.")
    print(f"Média das features - antes: {np.mean(X_train, axis=0)[:3]}...")
    print(f"Média das features - depois: {np.mean(X_train_scaled, axis=0)[:3]}...")
    print(f"Desvio padrão - antes: {np.std(X_train, axis=0)[:3]}...")
    print(f"Desvio padrão - depois: {np.std(X_train_scaled, axis=0)[:3]}...")

    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessors


def save_preprocessors(
    preprocessors: Dict[str, Any], output_dir: str = "../models"
) -> None:
    """
    Salva os objetos de pré-processamento para uso posterior.

    Args:
        preprocessors (Dict[str, Any]): Dicionário contendo os objetos de pré-processamento.
        output_dir (str): Diretório onde os objetos serão salvos.
    """
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Salvar cada objeto de pré-processamento
    for name, obj in preprocessors.items():
        filepath = os.path.join(output_dir, f"{name}.pkl")
        joblib.dump(obj, filepath)
        print(f"\nObjeto de pré-processamento '{name}' salvo em {filepath}")


def load_preprocessors(input_dir: str = "../models") -> Dict[str, Any]:
    """
    Carrega os objetos de pré-processamento salvos anteriormente.

    Args:
        input_dir (str): Diretório onde os objetos estão salvos.

    Returns:
        Dict[str, Any]: Dicionário contendo os objetos de pré-processamento.
    """
    preprocessors = {}

    # Carregar cada objeto de pré-processamento
    for filename in os.listdir(input_dir):
        if filename.endswith(".pkl") and filename != "california_housing_mlp.pkl":
            name = filename.split(".")[0]
            filepath = os.path.join(input_dir, filename)
            preprocessors[name] = joblib.load(filepath)
            print(f"Objeto de pré-processamento '{name}' carregado de {filepath}")

    return preprocessors


def save_processed_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = "../data/processed",
) -> None:
    """
    Salva os dados pré-processados para uso posterior.

    Args:
        X_train, X_test, y_train, y_test: Arrays NumPy contendo os dados pré-processados.
        output_dir (str): Diretório onde os dados serão salvos.
    """
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Salvar os dados
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    print(f"Dados pré-processados salvos em {output_dir}")


def load_processed_data(
    input_dir: str = "../data/processed",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega os dados pré-processados salvos anteriormente.

    Args:
        input_dir (str): Diretório onde os dados estão salvos.

    Returns:
        Tuple contendo X_train, X_test, y_train, y_test.
    """
    # Carregar os dados
    X_train = np.load(os.path.join(input_dir, "X_train.npy"))
    X_test = np.load(os.path.join(input_dir, "X_test.npy"))
    y_train = np.load(os.path.join(input_dir, "y_train.npy"))
    y_test = np.load(os.path.join(input_dir, "y_test.npy"))

    print(f"Dados pré-processados carregados de {input_dir}")
    print(
        f"Shapes: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}"
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    # Script para executar o pré-processamento de forma independente
    print("Executando pré-processamento dos dados...\n")

    # Carregar dados
    df = load_california_housing_data()

    # Verificar estatísticas básicas
    print("\nEstatísticas básicas do dataset:\n")
    print(df.describe())

    # Pré-processar dados
    X_train, X_test, y_train, y_test, preprocessors = preprocess_data(df)

    # Salvar objetos de pré-processamento
    save_preprocessors(preprocessors)

    # Salvar dados pré-processados
    save_processed_data(X_train, X_test, y_train, y_test)

    print("\nPré-processamento concluído com sucesso!")
