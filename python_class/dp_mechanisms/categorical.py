import torch

def direct_encoding(categorical_data, epsilon):
    """
    Aplica codificación directa a datos categóricos utilizando PyTorch.

    Parámetros:
    - categorical_data (torch.Tensor o array-like): Datos categóricos de entrada.
    - epsilon (float): Presupuesto de privacidad.

    Devuelve:
    - torch.Tensor: Datos categóricos transformados con privacidad diferencial.
    """
    # Convertir los datos de entrada a un tensor de PyTorch si no lo es
    if not isinstance(categorical_data, torch.Tensor):
        categorical_data = torch.tensor(categorical_data)
    
    # Obtener valores únicos y asignar índices inversos
    unique_values, inverse_indices = torch.unique(categorical_data, return_inverse=True)
    k = len(unique_values)  # Número de categorías únicas

    # Calcular probabilidades p y q
    e_epsilon = torch.exp(torch.tensor(epsilon, dtype=torch.float32))
    p = e_epsilon / (e_epsilon + k - 1)
    q = 1 / (e_epsilon + k - 1)

    # Crear matriz de probabilidades inicializada con q
    probabilities = torch.full((len(categorical_data), k), q)
    # Asignar la probabilidad p a la categoría original de cada dato
    probabilities[torch.arange(len(categorical_data)), inverse_indices] = p

    # Generar valores aleatorios uniformes entre 0 y 1
    random_vals = torch.rand(len(categorical_data), dtype=torch.float32)
    # Calcular los umbrales acumulados para cada categoría
    thresholds = torch.cumsum(probabilities, dim=1)
    # Determinar el índice de la categoría privatizada comparando los valores aleatorios con los umbrales
    privatized_indices = torch.sum(random_vals.unsqueeze(1) > thresholds, dim=1)

    # Convertir los índices privatizados a los valores categóricos correspondientes
    privatized_data = unique_values[privatized_indices]
    return privatized_data

def optimized_unary_encoding(categorical_data, epsilon):
    """
    Aplica Optimized Unary Encoding (OUE) a datos categóricos utilizando PyTorch.

    Parámetros:
    - categorical_data (torch.Tensor o array-like): Datos categóricos de entrada.
    - epsilon (float): Presupuesto de privacidad.

    Devuelve:
    - torch.Tensor: Datos categóricos transformados con privacidad diferencial.
    """
    # Convertir los datos de entrada a un tensor de PyTorch si no lo es
    if not isinstance(categorical_data, torch.Tensor):
        categorical_data = torch.tensor(categorical_data)
    
    # Obtener valores únicos y asignar índices inversos
    unique_values, inverse_indices = torch.unique(categorical_data, return_inverse=True)
    d = len(unique_values)  # Número de categorías únicas

    # Definir probabilidades p y q
    p = 0.5
    e_epsilon = torch.exp(torch.tensor(epsilon, dtype=torch.float32))
    q = 1 / (e_epsilon + 1)

    # Crear matriz binaria original (One-Hot Encoding)
    binary_matrix = torch.zeros((len(categorical_data), d), dtype=torch.float32)
    binary_matrix[torch.arange(len(categorical_data)), inverse_indices] = 1.0

    # Generar matriz perturbada según las probabilidades p y q
    random_matrix = torch.rand((len(categorical_data), d), dtype=torch.float32)
    perturbed_matrix = torch.where(
        binary_matrix == 1,
        (random_matrix < p).float(),
        (random_matrix < q).float()
    )

    # Seleccionar índices privatizados
    # Encontrar los índices donde la perturbed_matrix es 1
    positive_indices = torch.nonzero(perturbed_matrix, as_tuple=False)
    # Inicializar los índices privatizados con -1
    privatized_indices = torch.full((len(categorical_data),), -1, dtype=torch.long)
    # Para cada dato, seleccionar aleatoriamente una categoría positiva
    for idx in torch.unique(positive_indices[:, 0]):
        indices = positive_indices[positive_indices[:, 0] == idx, 1]
        privatized_indices[idx] = indices[torch.randint(len(indices), (1,))]
    # Manejar los casos donde no hay categorías positivas
    missing_indices = (privatized_indices == -1).nonzero(as_tuple=False).squeeze()
    if missing_indices.numel() > 0:
        privatized_indices[missing_indices] = torch.randint(d, (missing_indices.numel(),))

    # Convertir los índices privatizados a valores categóricos
    privatized_data = unique_values[privatized_indices]
    return privatized_data

def rappor(categorical_data, epsilon):
    """
    Aplica RAPPOR a datos categóricos utilizando PyTorch.

    Parámetros:
    - categorical_data (torch.Tensor o array-like): Datos categóricos de entrada.
    - epsilon (float): Presupuesto de privacidad.

    Devuelve:
    - torch.Tensor: Datos categóricos transformados con privacidad diferencial.
    """
    # Convertir los datos de entrada a un tensor de PyTorch si no lo es
    if not isinstance(categorical_data, torch.Tensor):
        categorical_data = torch.tensor(categorical_data)
    
    # Obtener valores únicos y asignar índices inversos
    unique_values, inverse_indices = torch.unique(categorical_data, return_inverse=True)
    d = len(unique_values)  # Número de categorías únicas

    # Calcular la probabilidad f
    f = 1 / (torch.exp(torch.tensor(epsilon, dtype=torch.float32)) + 1)

    # Crear matriz binaria original (One-Hot Encoding)
    binary_matrix = torch.zeros((len(categorical_data), d), dtype=torch.float32)
    binary_matrix[torch.arange(len(categorical_data)), inverse_indices] = 1.0

    # Generar matriz perturbada
    random_matrix = torch.rand((len(categorical_data), d), dtype=torch.float32)
    perturbed_matrix = torch.where(
        binary_matrix == 1,
        (random_matrix < (1 - f)).float(),
        (random_matrix < f).float()
    )

    # Seleccionar índices privatizados
    positive_indices = torch.nonzero(perturbed_matrix, as_tuple=False)
    privatized_indices = torch.full((len(categorical_data),), -1, dtype=torch.long)
    for idx in torch.unique(positive_indices[:, 0]):
        indices = positive_indices[positive_indices[:, 0] == idx, 1]
        privatized_indices[idx] = indices[torch.randint(len(indices), (1,))]
    # Manejar los casos donde no hay categorías positivas
    missing_indices = (privatized_indices == -1).nonzero(as_tuple=False).squeeze()
    if missing_indices.numel() > 0:
        privatized_indices[missing_indices] = torch.randint(d, (missing_indices.numel(),))

    # Convertir los índices privatizados a valores categóricos
    privatized_data = unique_values[privatized_indices]
    return privatized_data