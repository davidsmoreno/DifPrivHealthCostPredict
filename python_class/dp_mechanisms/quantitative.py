import numpy as np
import torch

def normalize_to_range(vector, min_range=-1, max_range=1):
    """
    Normaliza un vector de números a un rango especificado usando PyTorch.

    Parámetros:
    vector (list o np.array): El vector de números a normalizar.
    min_range (float): El valor mínimo del rango. Por defecto es -1.
    max_range (float): El valor máximo del rango. Por defecto es 1.

    Devuelve:
    tuple:
        normalized (torch.Tensor): El vector normalizado.
        min_val (float): El valor mínimo del vector original.
        max_val (float): El valor máximo del vector original.
    """
    vector = torch.tensor(vector, dtype=torch.float32)
    min_val = torch.min(vector)
    max_val = torch.max(vector)

    # Evitar división por cero si min_val == max_val
    if min_val == max_val:
        normalized = torch.zeros_like(vector)
    else:
        # Normalizar a [0, 1]
        normalized = (vector - min_val) / (max_val - min_val)
        # Escalar a [min_range, max_range]
        normalized = normalized * (max_range - min_range) + min_range

    return normalized, min_val.item(), max_val.item()

def denormalize_from_range(normalized_vector, min_val, max_val, min_range=-1, max_range=1):
    """
    Desnormaliza un vector desde un rango especificado de vuelta a su rango original.

    Parámetros:
    normalized_vector (torch.Tensor): El vector normalizado.
    min_val (float): El valor mínimo del vector original.
    max_val (float): El valor máximo del vector original.
    min_range (float): El valor mínimo del rango normalizado. Por defecto es -1.
    max_range (float): El valor máximo del rango normalizado. Por defecto es 1.

    Devuelve:
    torch.Tensor: El vector desnormalizado.
    """
    # Escalar de vuelta a [0, 1]
    normalized_vector = (normalized_vector - min_range) / (max_range - min_range)
    # Escalar al rango original [min_val, max_val]
    denormalized = normalized_vector * (max_val - min_val) + min_val
    return denormalized

def duchi_mechanism(t_i_vector, epsilon):
    """
    Aplica el mecanismo de Duchi et al. para privacidad diferencial a un vector numérico.

    Parámetros:
    t_i_vector (np.array): Vector de entrada de números en el rango [-1, 1].
    epsilon (float): Presupuesto de privacidad.

    Devuelve:
    np.array: Vector transformado con privacidad diferencial.
    """
    # Convertir el vector a tensor de PyTorch
    t_i_tensor = torch.from_numpy(t_i_vector.astype(np.float32))

    # Asegurar que los valores estén en el rango [-1, 1]
    t_i_tensor = torch.clamp(t_i_tensor, -1, 1)

    # Calcular e^epsilon
    e_epsilon = torch.exp(torch.tensor(epsilon, dtype=torch.float32))

    # Calcular la probabilidad p para la respuesta aleatoria
    prob = 0.5 * (t_i_tensor + 1) * (e_epsilon / (e_epsilon + 1)) + \
           (1 - (t_i_tensor + 1) / 2) * (1 / (e_epsilon + 1))

    # Generar valores aleatorios según la distribución Bernoulli
    u = torch.bernoulli(prob)

    # Calcular el valor privatizado
    t_i_star = (2 * u - 1) * (e_epsilon + 1) / (e_epsilon - 1)

    # Convertir el tensor a un arreglo de NumPy y devolverlo
    return t_i_star.numpy()

def piecewise_mechanism(t_i_vector, epsilon):
    """
    Aplica el Mecanismo por Tramos para datos numéricos unidimensionales.

    Parámetros:
    t_i_vector (np.array): Vector de entrada de números en el rango [-1, 1].
    epsilon (float): Presupuesto de privacidad.

    Devuelve:
    np.array: Vector transformado dentro del rango [-C, C].
    """
    # Convertir el vector a tensor de PyTorch
    t_i_tensor = torch.tensor(t_i_vector, dtype=torch.float32)

    # Asegurar que los valores estén en el rango [-1, 1]
    t_i_tensor = torch.clamp(t_i_tensor, -1.0, 1.0)

    # Calcular e^(epsilon/2) y C
    e_epsilon_2 = torch.exp(torch.tensor(epsilon / 2.0))
    C = (e_epsilon_2 + 1.0) / (e_epsilon_2 - 1.0)

    # Calcular l(t_i) y r(t_i)
    l = ((C + 1.0) / 2.0) * t_i_tensor - (C - 1.0) / 2.0
    r = l + C - 1.0

    # Generar valores aleatorios para la decisión
    x = torch.rand_like(t_i_tensor)
    threshold = e_epsilon_2 / (e_epsilon_2 + 1.0)
    rand_uniform = torch.rand_like(t_i_tensor)

    # Inicializar el tensor para los valores privatizados
    t_i_star = torch.empty_like(t_i_tensor)

    # Máscara para x < threshold
    mask1 = x < threshold
    low1 = l
    high1 = r
    high1 = torch.where(high1 <= low1, low1 + 1e-5, high1)
    t_i_star[mask1] = low1[mask1] + (high1[mask1] - low1[mask1]) * rand_uniform[mask1]

    # Máscara para x >= threshold
    mask2 = ~mask1
    interval_choice = torch.rand_like(t_i_tensor) < 0.5

    # Máscaras combinadas
    mask2_interval1 = mask2 & interval_choice  # Intervalo [-C, l(t_i)]
    mask2_interval2 = mask2 & ~interval_choice  # Intervalo [r(t_i), C]

    # Crear tensores de 'low' y 'high' para cada intervalo
    low2 = torch.empty_like(t_i_tensor)
    high2 = torch.empty_like(t_i_tensor)

    # Intervalo [-C, l(t_i)]
    low2[mask2_interval1] = -C
    high2[mask2_interval1] = l[mask2_interval1]
    high2[mask2_interval1] = torch.where(high2[mask2_interval1] <= low2[mask2_interval1],
                                         low2[mask2_interval1] + 1e-5,
                                         high2[mask2_interval1])
    t_i_star[mask2_interval1] = low2[mask2_interval1] + (high2[mask2_interval1] - low2[mask2_interval1]) * rand_uniform[mask2_interval1]

    # Intervalo [r(t_i), C]
    low2[mask2_interval2] = r[mask2_interval2]
    high2[mask2_interval2] = C
    high2[mask2_interval2] = torch.where(high2[mask2_interval2] <= low2[mask2_interval2],
                                         low2[mask2_interval2] + 1e-5,
                                         high2[mask2_interval2])
    t_i_star[mask2_interval2] = low2[mask2_interval2] + (high2[mask2_interval2] - low2[mask2_interval2]) * rand_uniform[mask2_interval2]

    # Convertir el tensor a un arreglo de NumPy y devolverlo
    return t_i_star.numpy()