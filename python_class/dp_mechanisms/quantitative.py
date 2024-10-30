import numpy as np
import torch

def duchi_mechanism(t_i_vector, epsilon):
    """
    Aplica el mecanismo de Duchi et al. para privacidad diferencial a un vector numérico.

    Parámetros:
    t_i_vector (np.array): Vector de entrada de números en el rango [-1, 1].
    epsilon (float): Parámetro de privacidad.

    Devuelve:
    np.array: Vector transformado con privacidad diferencial.
    """
    # Convertir el vector de entrada a un tensor de PyTorch
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
    epsilon (float): Parámetro de privacidad.

    Devuelve:
    np.array: Vector transformado dentro del rango [-C, C].
    """
    # Convertir el vector de entrada a un tensor de PyTorch
    t_i_tensor = torch.from_numpy(t_i_vector.astype(np.float32))

    # Asegurar que los valores estén en el rango [-1, 1]
    t_i_tensor = torch.clamp(t_i_tensor, -1, 1)

    # Calcular e^(epsilon/2) y C
    e_epsilon_2 = torch.exp(torch.tensor(epsilon / 2, dtype=torch.float32))
    C = (e_epsilon_2 + 1) / (e_epsilon_2 - 1)

    # Calcular los vectores l(t_i) y r(t_i)
    l = ((C + 1) / 2) * t_i_tensor - (C - 1) / 2
    r = l + C - 1

    # Generar valores aleatorios uniformes para la decisión
    x = torch.rand_like(t_i_tensor)

    # Calcular el umbral para la decisión
    threshold = e_epsilon_2 / (e_epsilon_2 + 1)

    # Generar valores aleatorios para la selección del intervalo
    rand_uniform = torch.rand_like(t_i_tensor)

    # Inicializar el tensor para los valores privatizados
    t_i_star = torch.empty_like(t_i_tensor)

    # Máscara para x < threshold
    mask = x < threshold
    low = l[mask]
    high = r[mask]
    # Asegurar que high > low
    high = torch.where(high <= low, low + 1e-5, high)
    t_i_star[mask] = low + (high - low) * rand_uniform[mask]

    # Máscara para x >= threshold
    mask = ~mask
    # Decidir entre el intervalo [-C, l(t_i)] o [r(t_i), C]
    interval_choice = torch.rand_like(t_i_tensor[mask]) < 0.5

    # Intervalo [-C, l(t_i)]
    low = -C
    high = l[mask]
    high = torch.where(high <= low, low + 1e-5, high)
    t_i_star[mask][interval_choice] = low + (high - low) * rand_uniform[mask][interval_choice]

    # Intervalo [r(t_i), C]
    low = r[mask]
    high = C
    high = torch.where(high <= low, low + 1e-5, high)
    t_i_star[mask][~interval_choice] = low + (high - low) * rand_uniform[mask][~interval_choice]

    # Convertir el tensor a un arreglo de NumPy y devolverlo
    return t_i_star.numpy()