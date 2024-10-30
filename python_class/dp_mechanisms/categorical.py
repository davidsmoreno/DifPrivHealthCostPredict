import numpy as np

def direct_encoding(categorical_data, epsilon):
    """
    Aplica codificación directa (Respuesta Aleatoria) a datos categóricos.

    Parámetros:
    categorical_data (np.array): Datos categóricos de entrada.
    epsilon (float): Parámetro de privacidad.

    Devuelve:
    np.array: Datos categóricos transformados.
    """
    # Obtener los valores únicos y asignar índices
    unique_values, inverse_indices = np.unique(categorical_data, return_inverse=True)
    k = len(unique_values)

    # Calcular las probabilidades p y q
    e_epsilon = np.exp(epsilon)
    p = e_epsilon / (e_epsilon + k - 1)
    q = 1 / (e_epsilon + k - 1)

    # Crear una matriz de probabilidades
    probabilities = np.full((len(categorical_data), k), q)
    probabilities[np.arange(len(categorical_data)), inverse_indices] = p

    # Muestrear los índices privatizados utilizando operaciones vectorizadas
    random_vals = np.random.rand(len(categorical_data))
    thresholds = np.cumsum(probabilities, axis=1)
    privatized_indices = np.sum(random_vals[:, None] > thresholds, axis=1)

    # Convertir los índices privatizados a valores categóricos
    privatized_data = unique_values[privatized_indices]
    return privatized_data

def optimized_unary_encoding(categorical_data, epsilon):
    """
    Aplica Optimized Unary Encoding (OUE) a datos categóricos.

    Parámetros:
    categorical_data (np.array): Datos categóricos de entrada.
    epsilon (float): Parámetro de privacidad.

    Devuelve:
    np.array: Datos categóricos transformados.
    """
    # Obtener los valores únicos y asignar índices
    unique_values, inverse_indices = np.unique(categorical_data, return_inverse=True)
    d = len(unique_values)

    # Definir las probabilidades p y q
    p = 0.5
    e_epsilon = np.exp(epsilon)
    q = 1 / (e_epsilon + 1)

    # Crear la matriz binaria original
    binary_matrix = np.zeros((len(categorical_data), d), dtype=np.int8)
    binary_matrix[np.arange(len(categorical_data)), inverse_indices] = 1

    # Generar la matriz perturbada
    random_matrix = np.random.rand(len(categorical_data), d)
    perturbed_matrix = np.where(
        binary_matrix == 1,
        random_matrix < p,
        random_matrix < q
    )

    # Seleccionar índices privatizados
    positive_indices = np.argwhere(perturbed_matrix)
    # Inicializar índices privatizados con -1
    privatized_indices = np.full(len(categorical_data), -1, dtype=int)
    # Para filas con bits positivos, elegir uno al azar
    for idx in np.unique(positive_indices[:, 0]):
        indices = positive_indices[positive_indices[:, 0] == idx, 1]
        privatized_indices[idx] = np.random.choice(indices)
    # Para filas sin bits positivos, seleccionar un índice aleatorio
    missing_indices = np.where(privatized_indices == -1)[0]
    if len(missing_indices) > 0:
        privatized_indices[missing_indices] = np.random.randint(d, size=len(missing_indices))

    # Convertir los índices privatizados a valores categóricos
    privatized_data = unique_values[privatized_indices]
    return privatized_data

def rappor(categorical_data, epsilon):
    """
    Aplica RAPPOR (Randomized Aggregatable Privacy-Preserving Ordinal Response) a datos categóricos.

    Parámetros:
    categorical_data (np.array): Datos categóricos de entrada.
    epsilon (float): Parámetro de privacidad.

    Devuelve:
    np.array: Datos categóricos transformados.
    """
    # Obtener los valores únicos y asignar índices
    unique_values, inverse_indices = np.unique(categorical_data, return_inverse=True)
    d = len(unique_values)

    # Calcular la probabilidad f basada en epsilon
    f = 1 / (np.exp(epsilon) + 1)

    # Crear la matriz binaria original
    binary_matrix = np.zeros((len(categorical_data), d), dtype=np.int8)
    binary_matrix[np.arange(len(categorical_data)), inverse_indices] = 1

    # Generar la matriz perturbada
    random_matrix = np.random.rand(len(categorical_data), d)
    perturbed_matrix = np.where(
        binary_matrix == 1,
        random_matrix < (1 - f),
        random_matrix < f
    )

    # Seleccionar índices privatizados
    positive_indices = np.argwhere(perturbed_matrix)
    # Inicializar índices privatizados con -1
    privatized_indices = np.full(len(categorical_data), -1, dtype=int)
    # Para filas con bits positivos, elegir uno al azar
    for idx in np.unique(positive_indices[:, 0]):
        indices = positive_indices[positive_indices[:, 0] == idx, 1]
        privatized_indices[idx] = np.random.choice(indices)
    # Para filas sin bits positivos, seleccionar un índice aleatorio
    missing_indices = np.where(privatized_indices == -1)[0]
    if len(missing_indices) > 0:
        privatized_indices[missing_indices] = np.random.randint(d, size=len(missing_indices))

    # Convertir los índices privatizados a valores categóricos
    privatized_data = unique_values[privatized_indices]
    return privatized_data