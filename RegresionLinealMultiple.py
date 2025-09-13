import numpy as np
import pandas as pd
from scipy.stats import t, f


def numero_datos(X: pd.DataFrame) -> int:
    """
    Obtiene el número de observaciones en los datos.

    :param X: DataFrame con las variables independientes.
    :type X: pd.DataFrame

    :return: Número de observaciones.
    :rtype: int
    """
    return len(X)


def numero_parametros(X: pd.DataFrame, incluir_intercepto=True) -> int:
    """
    Obtiene el número de parámetros en el modelo.

    :param X: DataFrame con las variables independientes.
    :type X: pd.DataFrame
    :param incluir_intercepto: Si True, agrega 1 al conteo para el intercepto.

    :return: Número de parámetros.
    :rtype: int
    """
    return len(X.columns) + (1 if incluir_intercepto else 0)


def numero_variables(X: pd.DataFrame) -> int:
    """
    Obtiene el número de variables independientes en los datos.

    :param X: DataFrame con las variables independientes.
    :type X: pd.DataFrame

    :return: Número de variables independientes.
    :rtype: int
    """
    return len(X.columns)


def codificar_variables_categoricas(X: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica las variables categóricas en la matriz X utilizando codificación one-hot
    y agrega interacciones entre las variables categóricas y las variables explicativas.

    :param X: DataFrame con las variables independientes.
    :type X: pd.DataFrame

    :return: DataFrame con las variables categóricas codificadas y las interacciones.
    :rtype: pd.DataFrame
    """
    # Verificar si hay variables categóricas
    # Si no hay columnas categóricas, se retorna el DataFrame original
    if X.select_dtypes(include=['object', 'category']).empty:
        return X

    # Identificar las columnas categóricas en el DataFrame
    columnas_categoricas = X.select_dtypes(include=['object', 'category']).columns

    # Codificar las variables categóricas utilizando one-hot encoding
    # Esto crea nuevas columnas binarias para cada categoría
    X_codificado = pd.get_dummies(X, columns=columnas_categoricas)

    # Identificar las columnas numéricas en el DataFrame
    columnas_numericas = X.select_dtypes(include=['number']).columns

    # Iterar sobre cada columna categórica para generar interacciones
    for columna_categorica in columnas_categoricas:
        # Crear las columnas codificadas para la variable categórica actual
        columnas_codificadas = pd.get_dummies(X[columna_categorica], prefix=columna_categorica)

        # Iterar sobre cada columna numérica para calcular las interacciones
        for columna_numerica in columnas_numericas:
            # Multiplicar las columnas codificadas por la columna numérica
            interacciones = columnas_codificadas.mul(X[columna_numerica], axis=0)

            # Renombrar las columnas de las interacciones para reflejar las combinaciones
            interacciones.columns = [f"{columna_codificada}*{columna_numerica}" for columna_codificada in columnas_codificadas.columns]

            # Agregar las columnas de interacciones al DataFrame codificado
            X_codificado = pd.concat([X_codificado, interacciones], axis=1)

    # Retornar el DataFrame con las variables categóricas codificadas y las interacciones
    return X_codificado


def calcular_estimadores(X: pd.DataFrame, y: pd.Series, incluir_intercepto=True) -> pd.Series:
    """
    Calcula los estimadores de una regresión múltiple con OLS.

    :param X: DataFrame con las variables independientes.
    :type X: pd.DataFrame
    :param y: Serie o DataFrame con la variable dependiente (n x 1).
    :type y: pd.Series o pd.DataFrame
    :param incluir_intercepto: Si True, agrega una columna de unos a X para estimar el intercepto.

    :return: Vector de coeficientes estimados (p x 1).
    :rtype: pd.Series
    """
    # Convertir a matriz numpy
    X_matrix = X.to_numpy()
    y_vector = y.to_numpy().reshape(-1, 1)  # asegurar columna

    # Agregar columna de 1's si se desea intercepto
    if incluir_intercepto:
        X_matrix = np.column_stack((np.ones(X_matrix.shape[0]), X_matrix))

    # Fórmula de OLS: (X^T X)^(-1) X^T y
    estimadores = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_vector

    return pd.Series(estimadores)


def calcular_varianza(X: pd.DataFrame, Y: pd.Series, estimadores: pd.Series, incluir_intercepto=True, n=None, p=None) -> float:
    """
    Calcula la varianza explicada y la varianza residual de un modelo de regresión.

    :param X: DataFrame con las variables independientes.
    :type X: pd.DataFrame
    :param y: Serie o DataFrame con la variable dependiente (n x 1).
    :type y: pd.Series o pd.DataFrame
    :param estimadores: Coeficientes estimados del modelo (p x 1).
    :type estimadores: np.ndarray
    :param incluir_intercepto: Si True, incluye el intercepto en el cálculo.
    :type incluir_intercepto: bool
    :param n: Número de observaciones. Si None, se calcula como el número de filas en X.
    :type n: int
    :param p: Número de variables independientes. Si None, se calcula como el número de columnas en X.
    :type p: int

    :return: varianza
    :rtype: float
    """
    # Convertir el DataFrame X y estimadores a matrices numpy
    X_matrix = X.to_numpy()
    estimadores = estimadores.to_numpy()

    # Agregar una columna de 1's a X si se desea incluir el intercepto en el modelo
    if incluir_intercepto:
        X_matrix = np.column_stack((np.ones(X_matrix.shape[0]), X_matrix))

    # Si no se proporciona el número de observaciones (n), se calcula como el número de filas en X
    if n is None:
        n = numero_datos(X)

    # Si no se proporciona el número de parámetros (p), se calcula como el número de columnas en X
    if p is None:
        p = numero_parametros(X, incluir_intercepto)

    # Calcular la varianza residual usando la fórmula:
    # varianza = (Y - X * estimadores)^T * (Y - X * estimadores) / (n - p)
    sse = suma_cuadrada_error(Y, pd.Series(X_matrix @ estimadores))
    varianza = media_cuadratica_error(sse, n, p)

    return varianza


def matriz_covarianza_estimadores(X: pd.DataFrame, varianza: float) -> pd.DataFrame:
    """
    Calcula la matriz de covarianza de un DataFrame.

    :param X: DataFrame con las variables.
    :type X: pd.DataFrame
    :param varianza: Varianza de la variable dependiente.
    :type varianza: float

    :return: Matriz de covarianza (p x p).
    :rtype: pd.DataFrame
    """
    # Convertir el DataFrame X a una matriz numpy
    X_matrix = X.to_numpy()

    # Calcular la matriz de covarianza de los coeficientes estimadores
    # Fórmula: varianza * (X^T X)^(-1)
    matriz_covarianza = varianza * np.linalg.inv(X_matrix.T @ X_matrix)
    
    # Retornar la matriz de covarianza calculada
    return pd.DataFrame(matriz_covarianza)


def prueba_significancia_individual(estimadores: pd.Series, matriz_covarianza: pd.DataFrame, n: int, p=None, nivel_significancia=0.05) -> pd.DataFrame:
    """
    Realiza la prueba de significancia individual para cada coeficiente estimador.

    :param estimadores: Coeficientes estimados del modelo (p x 1).
    :type estimadores: pd.Series
    :param matriz_covarianza: Matriz de covarianza de los coeficientes (p x p).
    :type matriz_covarianza: pd.DataFrame
    :param n: Número de observaciones.
    :type n: int
    :param nivel_significancia: Nivel de significancia para la prueba.
    :type nivel_significancia: float

    :return: Estadísticos t y p-valores para cada coeficiente.
    :rtype: pd.DataFrame
    """
    # Crear un DataFrame para almacenar los resultados de la prueba
    resultados = pd.DataFrame(columns=["Estadístico de prueba", "Valor critico", "Rechazo H0", "P-Value", "Nivel de significancia"])
    
    # Si no se proporciona el número de parámetros (p), se calcula como la longitud de los coeficientes estimadores
    if p is None:
        p = len(estimadores)
    
    # Calcular el valor crítico t basado en el nivel de significancia y los grados de libertad
    valor_critico = t.ppf(1 - nivel_significancia / 2, df=(n - p)) # Es lo mismo que una distribucion F con 1 y n-p grados de libertad

    # Iterar sobre cada coeficiente estimador para realizar la prueba de significancia
    for j in range(len(estimadores)):
        # Calcular el estadístico de prueba t para el coeficiente estimador j
        estadistico_prueba = estimadores[j] / np.sqrt(matriz_covarianza[j, j])

        # Determinar si se rechaza la hipótesis nula (H0) para el coeficiente estimador j
        rechazo_h0 = abs(estadistico_prueba) > valor_critico
        
        # Calcular el p-valor asociado al estadístico t
        p_value = 2 * (1 - t.cdf(abs(estadistico_prueba), df=(n - p)))

        # Almacenar los resultados en el DataFrame
        resultados.loc[j] = [estadistico_prueba, valor_critico, rechazo_h0, p_value, nivel_significancia]

    # Retornar el DataFrame con los resultados de la prueba
    return resultados


def intervalo_confianza(varianza: float, x_particular: pd.Series, estimadores: pd.Series, X: pd.DataFrame, incluir_intercepto=True, n=None, p=None, nivel_significancia=0.05) -> dict:
    """
    Calcula el intervalo de confianza para los coeficientes estimadores.

    :param estimadores: Coeficientes del modelo (p x 1).
    :type estimadores: pd.Series
    :param matriz_covarianza: Matriz de covarianza de los coeficientes (p x p).
    :type matriz_covarianza: pd.DataFrame
    :param varianza: Varianza del modelo.
    :type varianza: float
    :param n: Número de observaciones.
    :type n: int
    :param p: Número de parámetros (incluyendo el intercepto).
    :type p: int
    :param nivel_significancia: Nivel de significancia para el intervalo de confianza.
    :type nivel_significancia: float

    :return: Intervalos de confianza para cada coeficiente (p x 2).
    :rtype: pd.DataFrame
    """
    # Convertir a matriz numpy
    X_matrix = X.to_numpy()
    estimadores = estimadores.to_numpy()
    x_particular = x_particular.to_numpy()
    
    # Agregar una columna de 1's a X si se desea incluir el intercepto en el modelo
    if incluir_intercepto:
        X_matrix = np.column_stack((np.ones(X_matrix.shape[0]), X_matrix))

    # Si no se proporciona el número de parámetros (p), se calcula como la longitud de los coeficientes estimadores
    if p is None:
        p = len(estimadores)

    # Si no se proporciona el número de observaciones (n), se calcula como la longitud de X
    if n is None:
        n = numero_datos(X)

    # Calcular el valor crítico t basado en el nivel de significancia y los grados de libertad
    valor_critico = t.ppf(1 - nivel_significancia / 2, df=(n - p))

    # Calcular el margen de error para el intervalo de predicción
    margen_error = valor_critico * np.sqrt(varianza * x_particular.T @ np.linalg.inv(X_matrix.T @ X_matrix) @ x_particular)

    # Calcular los límites del intervalo de predicción
    limite_superior = (x_particular @ estimadores) + margen_error
    limite_inferior = (x_particular @ estimadores) - margen_error

    return {'Limite Inferior': limite_inferior, 'Limite Superior': limite_superior}


def intervalo_prediccion(varianza: float, x_particular: pd.Series, estimadores: pd.Series, X: pd.DataFrame, incluir_intercepto=True, n=None, p=None, nivel_significancia=0.05) -> dict:
    """
    Calcula el intervalo de predicción para unos valores específicos.

    :param x_particular: Valores de las características para el nuevo punto.
    :type x_particular: pd.Series
    :param estimadores: Coeficientes del modelo (p x 1).
    :type estimadores: pd.Series
    :param varianza: Varianza del modelo.
    :type varianza: float
    :param n: Número de observaciones.
    :type n: int
    :param p: Número de parámetros (incluyendo el intercepto).
    :type p: int
    :param nivel_significancia: Nivel de significancia para el intervalo de confianza.
    :type nivel_significancia: float

    :return: Intervalo de predicción.
    :rtype: dict
    """
    
    # Convertir a matriz numpy
    X_matrix = X.to_numpy()
    x_particular = x_particular.to_numpy().reshape(-1, 1)  # Asegurar que sea un vector columna
    
    # Agregar una columna de 1's a X si se desea incluir el intercepto en el modelo
    if incluir_intercepto:
        X_matrix = np.column_stack((np.ones(X_matrix.shape[0]), X_matrix))

    # Si no se proporciona el número de parámetros (p), se calcula como la longitud de los coeficientes estimadores
    if p is None:
        p = len(estimadores)

    # Si no se proporciona el número de observaciones (n), se calcula como la longitud de X
    if n is None:
        n = numero_datos(X)

    # Calcular el valor crítico t basado en el nivel de significancia y los grados de libertad
    valor_critico = t.ppf(1 - nivel_significancia / 2, df=(n - p))

    # Calcular el margen de error para el intervalo de predicción
    margen_error = valor_critico * np.sqrt(varianza * x_particular.T @ np.linalg.inv(X_matrix.T @ X_matrix) @ x_particular + varianza)

    # Calcular los límites del intervalo de predicción
    limite_superior = (x_particular @ estimadores) + margen_error
    limite_inferior = (x_particular @ estimadores) - margen_error

    return {'Limite Inferior': limite_inferior, 'Limite Superior': limite_superior}


def suma_cuadrada_regresion(Y_pred: pd.Series, promedio: float) -> float:
    """
    Calcula la suma de cuadrados de la regresión.

    :param Y: Valores estimados de la variable dependiente.
    :type Y: pd.Series
    :param promedio: Promedio de los valores reales de la variable dependiente.
    :type promedio: float

    :return: Suma de cuadrados de la regresión.
    :rtype: float
    """
    return np.sum((Y_pred - promedio) ** 2)


def suma_cuadrada_error(Y: pd.Series, Y_pred: pd.Series) -> float:
    """
    Calcula la suma de cuadrados del error.

    :param Y: Valores reales de la variable dependiente.
    :type Y: pd.Series
    :param Y_pred: Valores predichos de la variable dependiente.
    :type Y_pred: pd.Series

    :return: Suma de cuadrados del error.
    :rtype: float
    """
    return np.sum((Y - Y_pred) ** 2)


def suma_cuadrada_total(Y: pd.Series, promedio: float) -> float:
    """
    Calcula la suma de cuadrados total.

    :param Y: Valores reales de la variable dependiente.
    :type Y: pd.Series
    :param promedio: Promedio de los valores reales de la variable dependiente.
    :type promedio: float

    :return: Suma de cuadrados total.
    :rtype: float
    """
    return np.sum((Y - promedio) ** 2)


def media_cuadratica_regresion(ssr: float, k: int) -> float:
    """
    Calcula la media cuadrática de la regresión.

    :param ssr: Suma de cuadrados de la regresión.
    :type ssr: float
    :param k: Número de variables independientes en el modelo.
    :type k: int

    :return: Media cuadrática de la regresión.
    :rtype: float
    """
    return ssr / k


def media_cuadratica_error(sse: float, n: int, p: int) -> float:
    """
    Calcula la media cuadrática del error.

    :param sse: Suma de cuadrados del error.
    :type sse: float
    :param n: Número de observaciones.
    :type n: int
    :param p: Número de parámetros en el modelo (incluyendo el intercepto).
    :type p: int

    :return: Media cuadrática del error.
    :rtype: float
    """
    return sse / (n - p)


def media_cuadratica_total(sst: float, n: int) -> float:
    """
    Calcula la media cuadrática total.

    :param sst: Suma de cuadrados total.
    :type sst: float
    :param n: Número de observaciones.
    :type n: int

    :return: Media cuadrática total.
    :rtype: float
    """
    return sst / (n - 1)


def prueba_significancia_global(msr: float, mse: float, k: int, n: int, p:int, nivel_significancia=0.05) -> pd.DataFrame:
    """
    Realiza la prueba de significancia global del modelo de regresión.

    :param msr: Media cuadrática de la regresión.
    :type msr: float
    :param mse: Media cuadrática del error.
    :type mse: float
    :param k: Número de variables independientes en el modelo.
    :type k: int
    :param n: Número de observaciones.
    :type n: int
    :param p: Número de parámetros en el modelo (incluyendo el intercepto).
    :type p: int
    :param nivel_significancia: Nivel de significancia para la prueba.
    :type nivel_significancia: float

    :return: Estadístico F, valor crítico y si se rechaza H0.
    :rtype: pd.DataFrame
    """
    # Crear un DataFrame para almacenar los resultados de la prueba
    resultados = pd.DataFrame(columns=["Estadístico de prueba", "Valor critico", "Rechazo H0", "P-Value", "Nivel de significancia"])
    # Calcular el estadístico de prueba
    estadistico_prueba = msr / mse

    # Calcular el valor crítico F basado en el nivel de significancia y los grados de libertad
    # Grados de libertad del numerador (k) y del denominador (n - p)
    valor_critico = f.ppf(1 - nivel_significancia, dfn=k, dfd=n - p)

    # Determinar si se rechaza la hipótesis nula (H0)
    rechazo_h0 = estadistico_prueba > valor_critico

    # Calcular el p-valor asociado al estadístico F
    p_value = 1 - f.cdf(estadistico_prueba, dfn=k, dfd=n - p)

    # Almacenar los resultados en el DataFrame
    resultados.loc[0] = [estadistico_prueba, valor_critico, rechazo_h0, p_value, nivel_significancia]
    return resultados


def anova(msr: float, mse: float, mst: float, k: int, n: int, p:int) -> pd.DataFrame:
    """
    Realiza el análisis de varianza (ANOVA) para un modelo de regresión.

    :param msr: Media cuadrática de la regresión.
    :type msr: float
    :param mse: Media cuadrática del error.
    :type mse: float
    :param mst: Media cuadrática total.
    :type mst: float
    :param k: Número de variables independientes en el modelo.
    :type k: int
    :param n: Número de observaciones.
    :type n: int
    :param p: Número de parámetros en el modelo (incluyendo el intercepto).
    :type p: int

    :return: Tabla ANOVA con fuentes de variación, sumas de cuadrados, grados de libertad, medias cuadráticas y estadísticos F.
    :rtype: pd.DataFrame
    """
    # Crear un DataFrame para la tabla ANOVA
    anova_table = pd.DataFrame(columns=["Fuente de Variación", "Suma de Cuadrados", "Grados de Libertad", "Media Cuadrática", "Estadístico F"])

    # Fuente de variación: Regresión
    anova_table.loc[0] = ["Regresión", msr * k, k, msr, msr / mse]

    # Fuente de variación: Error
    anova_table.loc[1] = ["Error", mse * (n - p), n - p, mse, np.nan]

    # Fuente de variación: Total
    anova_table.loc[2] = ["Total", mst * (n - 1), n - 1, mst, np.nan]

    return anova_table


def prueba_significancia_parcial(modelo_reducido:pd.Series, promedio: float, k_reducido:int, n:int, p:int, mse: float, nivel_significancia=0.05) -> pd.DataFrame:
    """
    Realiza la prueba de significancia parcial del modelo de regresión.

    :param modelo_reducido: Valores predichos del modelo reducido.
    :type modelo_reducido: pd.Series
    :param promedio: Promedio de los valores reales de la variable dependiente.
    :type promedio: float
    :param k_reducido: Número de variables independientes en el modelo reducido.
    :type k_reducido: int
    :param n: Número de observaciones.
    :type n: int
    :param p: Número de parámetros en el modelo completo (incluyendo el intercepto).
    :type p: int
    :param mse: Media cuadrática del error del modelo completo.
    :type mse: float
    :param nivel_significancia: Nivel de significancia para la prueba.
    :type nivel_significancia: float

    :return: Estadístico F, valor crítico, si se rechaza H0 y p-valor.
    :rtype: pd.DataFrame
    """
    resultados = pd.DataFrame(columns=["Estadístico de Prueba", "Valor critico", "Rechazo H0", "P-Value", "Nivel de significancia"])
    
    # Calcular la suma de cuadrados de la regresión del modelo reducido
    ssr_reducido = suma_cuadrada_regresion(modelo_reducido, promedio)
    
    # Calcular la media cuadrática de la regresión del modelo reducido
    msr_reducido = media_cuadratica_regresion(ssr_reducido, k_reducido)
    
    # Calcular el estadistico de prueba
    estadistico_prueba = msr_reducido / mse
    
    # Calcular el valor critico F basado en el nivel de significancia y los grados de libertad
    valor_critico = f.ppf(1 - nivel_significancia, dfn=k_reducido, dfd=n - p)
    
    # Determinar si se rechaza la hipótesis nula (H0)
    rechazo_h0 = estadistico_prueba > valor_critico
    
    # Calcular el p-valor asociado al estadístico F
    p_value = 1 - f.cdf(estadistico_prueba, dfn=k_reducido, dfd=n - p)
    
    # Almacenar los resultados en el DataFrame
    resultados.loc[0] = [estadistico_prueba, valor_critico, rechazo_h0, p_value, nivel_significancia]

    return resultados