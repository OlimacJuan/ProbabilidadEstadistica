import numpy as np
import pandas as pd


def calcular_betas(X: pd.DataFrame, y: pd.Series, incluir_intercepto=True) -> np.ndarray:
    """
    Calcula los coeficientes beta de una regresión múltiple con OLS.

    :param X: DataFrame con las variables independientes.
    :type X: pd.DataFrame
    :param y: Serie o DataFrame con la variable dependiente (n x 1).
    :type y: pd.Series o pd.DataFrame
    :param incluir_intercepto: Si True, agrega una columna de unos a X para estimar el intercepto.

    :return: Vector de coeficientes estimados (p x 1).
    :rtype: np.ndarray
    """

    # Convertir a matriz numpy
    X_matrix = X.to_numpy()
    y_vector = y.to_numpy().reshape(-1, 1)  # asegurar columna

    # Agregar columna de 1's si se desea intercepto
    if incluir_intercepto:
        X_matrix = np.column_stack((np.ones(X_matrix.shape[0]), X_matrix))

    # Fórmula de OLS: (X^T X)^(-1) X^T y
    betas = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_vector

    return betas


def calcular_varianza(X: pd.DataFrame, y: pd.Series, betas: np.ndarray, incluir_intercepto=True, n=None, p=None) -> float:
    """
    Calcula la varianza explicada y la varianza residual de un modelo de regresión.

    :param X: DataFrame con las variables independientes.
    :type X: pd.DataFrame
    :param y: Serie o DataFrame con la variable dependiente (n x 1).
    :type y: pd.Series o pd.DataFrame
    :param betas: Coeficientes estimados del modelo (p x 1).
    :type betas: np.ndarray
    :param incluir_intercepto: Si True, incluye el intercepto en el cálculo.
    :type incluir_intercepto: bool
    :param n: Número de observaciones. Si None, se calcula como el número de filas en X.
    :type n: int
    :param p: Número de variables independientes. Si None, se calcula como el número de columnas en X.
    :type p: int

    :return: varianza
    :rtype: float
    """
    X_matrix = X.to_numpy()
    y_vector = y.to_numpy().reshape(-1, 1)  # asegurar columna

    # Agregar columna de 1's si se desea intercepto
    if incluir_intercepto:
        X_matrix = np.column_stack((np.ones(X_matrix.shape[0]), X_matrix))

    # Si el número de observaciones (n) no se proporciona, se calcula como el número de filas en X
    if n is None:
        n = X_matrix.shape[0]

    # Si el número de parametros calculados (p) no se proporciona, se calcula como el número de columnas en la matriz X (incluyendo el intercepto si aplica)
    if p is None:
        p = X_matrix.shape[1]

    varianza = (y_vector - X_matrix @ betas).T @ (y_vector - X_matrix @ betas) / (n - p)

    return varianza