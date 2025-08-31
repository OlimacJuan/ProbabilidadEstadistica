import numpy as np
import pandas as pd
from scipy.stats import t 


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


def calcular_varianza(X: pd.DataFrame, Y: pd.Series, betas: np.ndarray, incluir_intercepto=True, n=None, p=None) -> float:
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
    # Convertir el DataFrame X y la Serie y a matrices numpy
    X_matrix = X.to_numpy()
    Y_vector = Y.to_numpy().reshape(-1, 1)  # Asegurar que y sea un vector columna

    # Agregar una columna de 1's a X si se desea incluir el intercepto en el modelo
    if incluir_intercepto:
        X_matrix = np.column_stack((np.ones(X_matrix.shape[0]), X_matrix))

    # Si no se proporciona el número de observaciones (n), se calcula como el número de filas en X
    if n is None:
        n = X_matrix.shape[0]

    # Si no se proporciona el número de parámetros (p), se calcula como el número de columnas en X
    if p is None:
        p = X_matrix.shape[1]

    # Calcular la varianza residual usando la fórmula:
    # varianza = (Y - X * betas)^T * (Y - X * betas) / (n - p)
    varianza = (Y_vector - X_matrix @ betas).T @ (Y_vector - X_matrix @ betas) / (n - p)

    return varianza


def matriz_covarianza_betas(X: pd.DataFrame, varianza: float) -> np.ndarray:
    """
    Calcula la matriz de covarianza de un DataFrame.

    :param X: DataFrame con las variables.
    :type X: pd.DataFrame
    :param varianza: Varianza de la variable dependiente.
    :type varianza: float

    :return: Matriz de covarianza (p x p).
    :rtype: np.ndarray
    """
    # Convertir el DataFrame X a una matriz numpy
    X_matrix = X.to_numpy()
    
    # Calcular la matriz de covarianza de los coeficientes beta
    # Fórmula: varianza * (X^T X)^(-1)
    matriz_covarianza = varianza * np.linalg.inv(X_matrix.T @ X_matrix)
    
    # Retornar la matriz de covarianza calculada
    return matriz_covarianza


def prueba_significancia_individual(betas: np.ndarray, matriz_covarianza: np.ndarray, n: int, p=None, nivel_significancia=0.05) -> np.ndarray:
    """
    Realiza la prueba de significancia individual para cada coeficiente beta.

    :param betas: Coeficientes estimados del modelo (p x 1).
    :type betas: np.ndarray
    :param matriz_covarianza: Matriz de covarianza de los coeficientes (p x p).
    :type matriz_covarianza: np.ndarray
    :param n: Número de observaciones.
    :type n: int
    :param nivel_significancia: Nivel de significancia para la prueba.
    :type nivel_significancia: float

    :return: Estadísticos t y p-valores para cada coeficiente.
    :rtype: np.ndarray
    """
    # Crear un DataFrame para almacenar los resultados de la prueba
    resultados = pd.DataFrame(columns=["Estadístico de prueba", "Valor critico", "Rechazo H0"])
    
    # Si no se proporciona el número de parámetros (p), se calcula como la longitud de los coeficientes beta
    if p is None:
        p = len(betas)
    
    # Calcular el valor crítico t basado en el nivel de significancia y los grados de libertad
    valor_critico = t.ppf(1 - nivel_significancia / 2, df=(n - p))

    # Iterar sobre cada coeficiente beta para realizar la prueba de significancia
    for j in range(len(betas)):
        # Calcular el estadístico de prueba t para el coeficiente beta j
        estadistico_prueba = betas[j] / np.sqrt(matriz_covarianza[j, j])
        
        # Determinar si se rechaza la hipótesis nula (H0) para el coeficiente beta j
        rechazo_h0 = abs(estadistico_prueba) > valor_critico

        # Almacenar los resultados en el DataFrame
        resultados.loc[j] = [estadistico_prueba, valor_critico, rechazo_h0]

    # Retornar el DataFrame con los resultados de la prueba
    return resultados


def intervalo_prediccion(varianza: float, n: int, x_particular: pd.Series, betas: np.ndarray, X: pd.DataFrame, incluir_intercepto=True, p=None, nivel_significancia=0.05) -> np.ndarray:
    """
    Calcula el intervalo de predicción para unos valores específicos.

    :param x_particular: Valores de las características para el nuevo punto.
    :type x_particular: pd.Series
    :param betas: Coeficientes del modelo (p x 1).
    :type betas: np.ndarray
    :param matriz_covarianza: Matriz de covarianza de los coeficientes (p x p).
    :type matriz_covarianza: np.ndarray
    :param varianza: Varianza del modelo.
    :type varianza: float
    :param n: Número de observaciones.
    :type n: int
    :param p: Número de parámetros (incluyendo el intercepto).
    :type p: int
    :param nivel_significancia: Nivel de significancia para el intervalo de confianza.
    :type nivel_significancia: float

    :return: Intervalo de predicción (inferior, superior).
    :rtype: np.ndarray
    """
    
    # Convertir a matriz numpy
    X_matrix = X.to_numpy()
    
    # Agregar una columna de 1's a X si se desea incluir el intercepto en el modelo
    if incluir_intercepto:
        X_matrix = np.column_stack((np.ones(X_matrix.shape[0]), X_matrix))
    
    # Si no se proporciona el número de parámetros (p), se calcula como la longitud de los coeficientes beta
    if p is None:
        p = len(betas)

    # Calcular el valor crítico t basado en el nivel de significancia y los grados de libertad
    valor_critico = t.ppf(1 - nivel_significancia / 2, df=(n - p))

    # Calcular la desviacion estandar de la prediccion
    desviacion_estandar = t.ppf(1 - nivel_significancia / 2, df=(n - p)) * np.sqrt(varianza * x_particular.T @ np.linalg.inv(X.T @ X) @ x_particular + varianza)

    # Calcular los límites del intervalo de predicción
    limite_superior = (x_particular @ betas) + desviacion_estandar
    limite_inferior = (x_particular @ betas) - desviacion_estandar

    return np.array([limite_inferior, limite_superior])