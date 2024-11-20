import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

def direct_encoding(categorical_data, epsilon):
    """
    Aplica codificación directa a datos categóricos usando PyTorch.

    Parámetros:
    - categorical_data (array-like): Datos categóricos de entrada.
    - epsilon (float): Presupuesto de privacidad.

    Devuelve:
    - np.array: Datos transformados con privacidad diferencial.
    """
    # Convertir los datos de entrada a un array de NumPy si no lo son
    if not isinstance(categorical_data, np.ndarray):
        categorical_data = np.array(categorical_data)

    # Asegurarse de que los datos son de tipo string (si no lo son ya)
    categorical_data = categorical_data.astype(str)

    # Codificar categorías en valores numéricos usando LabelEncoder
    le = LabelEncoder()
    categorical_data_encoded = le.fit_transform(categorical_data)
    inverse_indices = torch.tensor(categorical_data_encoded, dtype=torch.long)

    k = len(le.classes_)  # Número de categorías únicas

    # Calcular probabilidades p y q
    e_epsilon = torch.exp(torch.tensor(epsilon, dtype=torch.float32))
    p = e_epsilon / (e_epsilon + k - 1)
    q = 1 / (e_epsilon + k - 1)

    # Crear una matriz de probabilidades inicializada con q
    probabilities = torch.full((len(categorical_data), k), q)
    # Asignar probabilidad p a la categoría original de cada dato
    probabilities[torch.arange(len(categorical_data)), inverse_indices] = p

    # Generar valores aleatorios uniformes entre 0 y 1
    random_vals = torch.rand(len(categorical_data), dtype=torch.float32)
    # Calcular umbrales acumulativos para cada categoría
    thresholds = torch.cumsum(probabilities, dim=1)
    # Determinar el índice de categoría privatizado comparando valores aleatorios con umbrales
    privatized_indices = torch.sum(random_vals.unsqueeze(1) > thresholds, dim=1)

    # Mapear los índices privatizados de vuelta a las categorías originales
    privatized_data = le.inverse_transform(privatized_indices.numpy())
    return privatized_data

def optimized_unary_encoding(categorical_data, epsilon):
    """
    Aplica Codificación Unaria Optimizada (OUE) a datos categóricos usando PyTorch.

    Parámetros:
    - categorical_data (array-like): Datos categóricos de entrada.
    - epsilon (float): Presupuesto de privacidad.

    Devuelve:
    - np.array: Datos transformados con privacidad diferencial.
    """
    # Convertir los datos de entrada a un array de NumPy si no lo son
    if not isinstance(categorical_data, np.ndarray):
        categorical_data = np.array(categorical_data)
    
    # Asegurarse de que los datos son de tipo string (si no lo son ya)
    categorical_data = categorical_data.astype(str)
    
    # Codificar categorías en valores numéricos usando LabelEncoder
    le = LabelEncoder()
    categorical_data_encoded = le.fit_transform(categorical_data)
    
    # Convertir los datos codificados a un tensor de PyTorch
    categorical_data_tensor = torch.tensor(categorical_data_encoded, dtype=torch.long)
    
    # Obtener valores únicos e índices inversos
    unique_values = torch.unique(categorical_data_tensor)
    inverse_indices = categorical_data_tensor
    d = len(unique_values)  # Número de categorías únicas
    
    # Definir probabilidades p y q
    p = 0.5
    e_epsilon = torch.exp(torch.tensor(epsilon, dtype=torch.float32))
    q = 1 / (e_epsilon + 1)
    
    # Crear matriz binaria original (Codificación One-Hot)
    binary_matrix = torch.zeros((len(categorical_data_tensor), d), dtype=torch.float32)
    binary_matrix[torch.arange(len(categorical_data_tensor)), inverse_indices] = 1.0
    
    # Generar matriz perturbada según probabilidades p y q
    random_matrix = torch.rand((len(categorical_data_tensor), d), dtype=torch.float32)
    perturbed_matrix = torch.where(
        binary_matrix == 1,
        (random_matrix < p).float(),
        (random_matrix < q).float()
    )
    
    # Seleccionar índices privatizados
    # Encontrar índices donde perturbed_matrix es 1
    positive_indices = torch.nonzero(perturbed_matrix, as_tuple=False)
    
    # Inicializar índices privatizados con -1
    privatized_indices = torch.full((len(categorical_data_tensor),), -1, dtype=torch.long)
    
    # Para cada punto de datos, seleccionar aleatoriamente una categoría positiva
    for idx in torch.unique(positive_indices[:, 0]):
        indices = positive_indices[positive_indices[:, 0] == idx, 1]
        if len(indices) > 0:
            random_choice = torch.randint(len(indices), (1,))
            privatized_indices[idx] = indices[random_choice]
    
    # Manejar casos donde no hay categorías positivas
    missing_indices = (privatized_indices == -1).nonzero(as_tuple=False).squeeze()
    if missing_indices.numel() > 0:
        privatized_indices[missing_indices] = torch.randint(d, (missing_indices.numel(),))
    
    # Convertir índices privatizados de vuelta a categorías originales
    privatized_data_encoded = privatized_indices.numpy()
    privatized_data = le.inverse_transform(privatized_data_encoded)
    
    return privatized_data

def rappor(categorical_data, epsilon):
    """
    Aplica RAPPOR a datos categóricos usando PyTorch.

    Parámetros:
    - categorical_data (array-like): Datos categóricos de entrada.
    - epsilon (float): Presupuesto de privacidad.

    Devuelve:
    - np.array: Datos transformados con privacidad diferencial.
    """
    # Convertir los datos de entrada a un array de NumPy si no lo son
    if not isinstance(categorical_data, np.ndarray):
        categorical_data = np.array(categorical_data)
    
    # Asegurarse de que los datos son de tipo string
    categorical_data = categorical_data.astype(str)
    
    # Codificar categorías en valores numéricos usando LabelEncoder
    le = LabelEncoder()
    categorical_data_encoded = le.fit_transform(categorical_data)
    
    # Convertir los datos codificados a un tensor de PyTorch
    categorical_data_tensor = torch.tensor(categorical_data_encoded, dtype=torch.long)
    
    # Obtener valores únicos e índices inversos
    unique_values = torch.unique(categorical_data_tensor)
    inverse_indices = categorical_data_tensor
    d = len(unique_values)  # Número de categorías únicas

    # Calcular la probabilidad f
    e_epsilon = torch.exp(torch.tensor(epsilon, dtype=torch.float32))
    f = 1 / (e_epsilon + 1)

    # Crear matriz binaria original (Codificación One-Hot)
    binary_matrix = torch.zeros((len(categorical_data_tensor), d), dtype=torch.float32)
    binary_matrix[torch.arange(len(categorical_data_tensor)), inverse_indices] = 1.0

    # Generar matriz perturbada
    random_matrix = torch.rand((len(categorical_data_tensor), d), dtype=torch.float32)
    perturbed_matrix = torch.where(
        binary_matrix == 1,
        (random_matrix < (1 - f)).float(),
        (random_matrix < f).float()
    )

    # Seleccionar índices privatizados
    positive_indices = torch.nonzero(perturbed_matrix, as_tuple=False)
    privatized_indices = torch.full((len(categorical_data_tensor),), -1, dtype=torch.long)
    for idx in torch.unique(positive_indices[:, 0]):
        indices = positive_indices[positive_indices[:, 0] == idx, 1]
        if len(indices) > 0:
            random_choice = torch.randint(len(indices), (1,))
            privatized_indices[idx] = indices[random_choice]
    
    # Manejar casos donde no hay categorías positivas
    missing_indices = (privatized_indices == -1).nonzero(as_tuple=False).squeeze()
    if missing_indices.numel() > 0:
        privatized_indices[missing_indices] = torch.randint(d, (missing_indices.numel(),))
    
    # Convertir índices privatizados de vuelta a categorías originales
    privatized_data_encoded = privatized_indices.numpy()
    privatized_data = le.inverse_transform(privatized_data_encoded)
    
    return privatized_data