import numpy as np
import pandas as pd
from scipy.stats import t 


def calcular_betas(X: pd.Series, Y: pd.Series, incluir_intercepto=True) -> np.ndarray:
    """
    Calcula los coeficientes beta de una regresión lineal simple.

    :param X: Serie con la variable independiente.
    :type X: pd.Series
    :param y: Serie con la variable dependiente.
    :type y: pd.Series
    :param incluir_intercepto: Si True, incluye el intercepto en el modelo.
    :type incluir_intercepto: bool

    :return: Coeficientes beta estimados.
    :rtype: np.ndarray
    """
    # Calcula la suma del producto de X e Y
    suma_producto = sum(Y[i] * X[i] for i in range(len(X)))
    
    # Obtiene el número de observaciones
    n = len(X)
    
    # Calcula los promedios de X e Y
    x_promedio = X.mean()
    y_promedio = Y.mean()
    
    # Calcula la suma de los cuadrados de X
    suma_cuadrada = sum(X[i]**2 for i in range(len(X)))

    # Calcula el coeficiente B_1 (pendiente)
    B_1 = (suma_producto - n * y_promedio * x_promedio) / (suma_cuadrada - n * x_promedio**2)
    
    # Calcula el coeficiente B_0 (intercepto)
    B_0 = y_promedio - B_1 * x_promedio

    # Retorna los coeficientes B_0 y B_1
    return np.array([B_0, B_1])


def calcular_varianza(X: pd.Series, Y: pd.Series, estimacion: pd.Series, n=None) -> float:
    """
    Calcula la varianza del modelo de regresión lineal simple.

    :param X: Serie con la variable independiente.
    :type X: pd.Series
    :param Y: Serie con la variable dependiente.
    :type Y: pd.Series
    :param estimacion: Serie con las estimaciones del modelo.
    :type estimacion: pd.Series

    :return: Varianza del modelo.
    :rtype: float
    """
    # Si no se proporciona el número de observaciones, se utiliza la longitud de X
    if n is None:
        n = len(X)
        
    # Calcula los residuales (diferencia entre los valores observados y estimados)
    residuales = Y - estimacion
    
    # Calcula la varianza dividiendo la suma de los cuadrados de los residuales
    # entre el número de grados de libertad (n - 2)
    varianza = sum(residuales**2) / (n - 2)
    
    # Retorna la varianza calculada
    return varianza


def matriz_covarianza_betas(X: pd.DataFrame, varianza: float) -> np.ndarray:
    """
    Calcula la matriz de covarianza de un DataFrame.

    :param X: DataFrame con las variables.
    :type X: pd.DataFrame
    :param varianza: Varianza de la variable dependiente.
    :type varianza: float

    :return: Matriz de covarianza (2 x 2).
    :rtype: np.ndarray
    """
    # Convertir el DataFrame X a una matriz numpy
    X_matrix = X.to_numpy()
    
    # Calcular la matriz de covarianza de los coeficientes beta
    # Fórmula: varianza * (X^T X)^(-1)
    matriz_covarianza = varianza * np.linalg.inv(X_matrix.T @ X_matrix)
    
    # Retornar la matriz de covarianza calculada
    return matriz_covarianza


def prueba_significancia_individual(betas: np.ndarray, matriz_covarianza: np.ndarray, n: int, nivel_significancia=0.05) -> pd.DataFrame:
    """
    Realiza la prueba de significancia individual para cada coeficiente beta.

    :param betas: Coeficientes estimados del modelo.
    :type betas: np.ndarray
    :param matriz_covarianza: Matriz de covarianza de los coeficientes (2 x 2).
    :type matriz_covarianza: np.ndarray
    :param n: Número de observaciones.
    :type n: int
    :param nivel_significancia: Nivel de significancia para la prueba.
    :type nivel_significancia: float

    :return: Resultados de la prueba de significancia.
    :rtype: pd.DataFrame
    """
    # Crear un DataFrame para almacenar los resultados de la prueba
    resultados = pd.DataFrame(columns=["Estadístico de prueba", "Valor critico", "Rechazo H0"])
    
    # Calcular el valor crítico t basado en el nivel de significancia y los grados de libertad
    valor_critico = t.ppf(1 - nivel_significancia / 2, df=(n - 2))

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


def intervalo_prediccion(varianza: float, n: int, x_particular: pd.Series, betas: np.ndarray, X: pd.DataFrame, incluir_intercepto=True, nivel_significancia=0.05) -> np.ndarray:
    """
    Calcula el intervalo de predicción para unos valores específicos.

    :param x_particular: Valores de las características para el nuevo punto.
    :type x_particular: pd.Series
    :param betas: Coeficientes del modelo.
    :type betas: np.ndarray
    :param matriz_covarianza: Matriz de covarianza de los coeficientes (2 x 2).
    :type matriz_covarianza: np.ndarray
    :param varianza: Varianza del modelo.
    :type varianza: float
    :param n: Número de observaciones.
    :type n: int
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

    # Calcular el valor crítico t basado en el nivel de significancia y los grados de libertad
    valor_critico = t.ppf(1 - nivel_significancia / 2, df=(n - 2))

    # Calcular la desviacion estandar de la prediccion
    desviacion_estandar = t.ppf(1 - nivel_significancia / 2, df=(n - 2)) * np.sqrt(varianza * x_particular.T @ np.linalg.inv(X.T @ X) @ x_particular + varianza)

    # Calcular los límites del intervalo de predicción
    limite_superior = (x_particular @ betas) + desviacion_estandar
    limite_inferior = (x_particular @ betas) - desviacion_estandar

    return np.array([limite_inferior, limite_superior])