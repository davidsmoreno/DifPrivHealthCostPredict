import numpy as np
import torch

def normalize_to_range(vector, min_range=-1.0, max_range=1.0):
    """
    Normaliza un vector de números al rango especificado usando PyTorch.

    Parámetros:
    - vector (array-like o torch.Tensor): El vector de números a normalizar.
    - min_range (float): El valor mínimo del rango destino. Por defecto es -1.0.
    - max_range (float): El valor máximo del rango destino. Por defecto es 1.0.

    Devuelve:
    - normalized (torch.Tensor): El vector normalizado.
    - min_val (float): El valor mínimo del vector original.
    - max_val (float): El valor máximo del vector original.
    """
    # Convertir el vector a tensor de PyTorch si no lo es
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector, dtype=torch.float32)
    else:
        vector = vector.clone().detach().float()  # Clonar para evitar modificar el original

    # Calcular el valor mínimo y máximo del vector
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
    - normalized_vector (torch.Tensor): El vector normalizado.
    - min_val (float): El valor mínimo del vector original.
    - max_val (float): El valor máximo del vector original.
    - min_range (float): El valor mínimo del rango normalizado. Por defecto es -1.
    - max_range (float): El valor máximo del rango normalizado. Por defecto es 1.

    Devuelve:
    - torch.Tensor: El vector desnormalizado.
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
    - t_i_vector (torch.Tensor o array-like): Vector de entrada de números, se asume en el rango [-1, 1].
    - epsilon (float): Presupuesto de privacidad (debe ser positivo).

    Devuelve:
    - t_i_star (torch.Tensor): Vector transformado con privacidad diferencial.
    """
    if epsilon <= 0:
        raise ValueError("Epsilon debe ser un valor positivo.")

    # Convertir el vector a tensor de doble precisión si no lo es
    if not isinstance(t_i_vector, torch.Tensor):
        t_i_tensor = torch.tensor(t_i_vector, dtype=torch.float64)
    else:
        t_i_tensor = t_i_vector.clone().detach().double()

    # Calcular tanh(epsilon / 2) para el escalado de privacidad
    epsilon_half = torch.tensor(epsilon / 2.0, dtype=torch.float64)
    tanh_epsilon_half = torch.tanh(epsilon_half)

    # Calcular la probabilidad p(x) para la muestra Bernoulli
    prob = 0.5 * (1.0 + t_i_tensor * tanh_epsilon_half)

    # Asegurarse de que las probabilidades estén dentro de [0, 1] para evitar problemas de muestreo
    prob = torch.clamp(prob, 0.0, 1.0)

    # Muestrear variable Bernoulli basada en la probabilidad
    u = torch.bernoulli(prob)

    # Calcular el factor de escalado
    w = 1.0 / tanh_epsilon_half

    # Aplicar la transformación privatizada
    t_i_star = (2 * u - 1.0) * w

    return t_i_star

def piecewise_mechanism(t_i_vector, epsilon):
    """
    Aplica el Mecanismo Por Partes para datos numéricos unidimensionales.

    Parámetros:
    - t_i_vector (torch.Tensor o np.array): Vector de entrada de números dentro del rango [-1, 1].
    - epsilon (float): Presupuesto de privacidad.

    Devuelve:
    - torch.Tensor: Vector transformado dentro del rango [-C, C].
    """
    # Asegurarse de que t_i_vector es un torch.Tensor
    if not isinstance(t_i_vector, torch.Tensor):
        t_i_tensor = torch.from_numpy(np.array(t_i_vector)).float()
    else:
        t_i_tensor = t_i_vector.float()

    # Asegurarse de que los valores estén dentro del rango [-1, 1]
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

    # Inicializar el tensor para valores privatizados
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

    # Crear tensores 'low' y 'high' para cada intervalo
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

    # Devolver el tensor
    return t_i_star

def laplace_mechanism(t_i_vector, epsilon, sensitivity=1.0):
    """
    Aplica el mecanismo de Laplace para privacidad diferencial a un vector numérico.

    Parámetros:
    - t_i_vector (torch.Tensor o array-like): Vector de entrada de números.
    - epsilon (float): Presupuesto de privacidad (debe ser positivo).
    - sensitivity (float): Sensibilidad del dato (debe ser positiva).

    Devuelve:
    - t_i_star (torch.Tensor): Vector transformado con privacidad diferencial utilizando ruido Laplaciano.
    """
    if epsilon <= 0:
        raise ValueError("Epsilon debe ser un valor positivo.")
    if sensitivity <= 0:
        raise ValueError("La sensibilidad debe ser un valor positivo.")

    # Convertir el vector a tensor si no lo es
    if not isinstance(t_i_vector, torch.Tensor):
        t_i_tensor = torch.tensor(t_i_vector, dtype=torch.float32)
    else:
        t_i_tensor = t_i_vector.clone().detach().float()

    # Calcular la escala del ruido de Laplace
    scale = sensitivity / epsilon

    # Generar ruido Laplaciano usando torch.distributions
    laplace_dist = torch.distributions.Laplace(loc=0.0, scale=scale)
    laplace_noise = laplace_dist.sample(t_i_tensor.shape)

    # Agregar el ruido Laplaciano al tensor original
    t_i_star = t_i_tensor + laplace_noise

    return t_i_star

def multidimensional_duchi_mechanism(t_i_vector, epsilon, B=1.0, num_samples=1000):
    """
    Aplica el mecanismo de Duchi et al. para privacidad diferencial a un vector multidimensional único con optimizaciones.

    Parámetros:
    - t_i_vector (torch.Tensor): Vector de entrada de forma (d,), con valores en el rango [-1, 1].
    - epsilon (float): Presupuesto de privacidad (debe ser positivo).
    - B (float): Valor máximo absoluto para los componentes de salida (por defecto es 1.0, debe ser positivo).
    - num_samples (int): Número de muestras para aproximar T+ y T− (reduce búsqueda exhaustiva).

    Devuelve:
    - t_i_star (torch.Tensor): Vector transformado con privacidad diferencial, forma (d,), valores en {-B, B}^d.
    """
    # Validar epsilon y B
    if epsilon <= 0:
        raise ValueError("Epsilon debe ser un valor positivo.")
    if B <= 0:
        raise ValueError("B debe ser un valor positivo.")

    # Asegurarse de que los valores en t_i_vector están dentro del rango [-1, 1]
    t_i_vector = torch.clamp(t_i_vector, -1.0, 1.0)
    d = t_i_vector.shape[0]  # Dimensión del vector

    # Paso 1: Generar vector aleatorio v ∈ {−1, 1}^d basado en probabilidades de t_i_vector
    prob_v = 0.5 * (1.0 + t_i_vector)  # Probabilidades para v[j] = 1
    random_vals_v = torch.rand(d)
    v = torch.where(random_vals_v < prob_v, torch.ones(d), -torch.ones(d))  # Vector v ∈ {−1, 1}^d

    # Paso 2: Muestrear T+ y T− con tuplas aleatorias en {-B, B}^d en lugar de generar todas las combinaciones
    T_plus = []
    T_minus = []

    for _ in range(num_samples):
        # Generar un candidato aleatorio en {-B, B}^d
        random_bits = torch.randint(0, 2, (d,))  # Generar bits aleatorios {0, 1} para cada componente
        t_candidate = torch.where(random_bits == 1, B, -B)  # Mapear bits a {-B, B}
        
        # Clasificar basado en producto punto con v
        if torch.dot(t_candidate, v) >= 0:
            T_plus.append(t_candidate)
        else:
            T_minus.append(t_candidate)

    # Convertir T_plus y T_minus a tensores para selección aleatoria eficiente
    T_plus = torch.stack(T_plus) if T_plus else None
    T_minus = torch.stack(T_minus) if T_minus else None

    # Paso 3: Muestrear una variable Bernoulli u que es 1 con probabilidad e^epsilon / (e^epsilon + 1)
    e_epsilon = torch.exp(torch.tensor(epsilon, dtype=torch.float32))
    prob_u = e_epsilon / (e_epsilon + 1.0)
    u = torch.bernoulli(torch.tensor(prob_u))

    # Paso 4: Seleccionar t_i_star de T+ o T− basado en el valor muestreado de u
    if u == 1 and T_plus is not None:
        # Seleccionar aleatoriamente de T+
        t_i_star = T_plus[torch.randint(len(T_plus), (1,)).item()]
    elif T_minus is not None:
        # Seleccionar aleatoriamente de T−
        t_i_star = T_minus[torch.randint(len(T_minus), (1,)).item()]
    else:
        # Respaldo en caso de que T_plus o T_minus estén vacíos debido al muestreo
        t_i_star = torch.full((d,), B if u == 1 else -B, dtype=torch.float32)

    return t_i_star

def multidimensional_mechanism(t_i_vector, epsilon, mechanism='piecewise', C=1.0):
    """
    Aplica un mecanismo de privacidad diferencial a un vector de una dimensión.

    Parámetros:
    - t_i_vector (torch.Tensor o array-like): Vector de entrada de forma (n_samples,), con valores en el rango [-1, 1].
    - epsilon (float): Presupuesto de privacidad (debe ser positivo).
    - mechanism (str): Mecanismo a utilizar ('piecewise', 'laplace' o 'duchi').
    - C (float): Constante para escalar los valores de salida (debe ser positivo, por defecto es 1.0).

    Devuelve:
    - t_i_star (torch.Tensor): Vector transformado con privacidad diferencial, de forma (n_samples,), con valores en el rango [-C, C].
    """
    if epsilon <= 0:
        raise ValueError("Epsilon debe ser un valor positivo.")
    if C <= 0:
        raise ValueError("C debe ser un valor positivo.")

    # Convertir la entrada a tensor si no lo es
    if not isinstance(t_i_vector, torch.Tensor):
        t_i_tensor = torch.tensor(t_i_vector, dtype=torch.float32)
    else:
        t_i_tensor = t_i_vector.clone().detach().float()

    # Asegurarse de que los valores están dentro del rango [-1, 1]
    t_i_tensor = torch.clamp(t_i_tensor, -1.0, 1.0)
    n_samples = t_i_tensor.shape[0]  # Número de muestras

    # Inicializar t_i_star como un tensor de ceros de forma (n_samples,)
    t_i_star = torch.zeros(n_samples, dtype=torch.float32)

    # Ajustar epsilon para cada muestra directamente
    epsilon_adjusted = epsilon

    # Aplicar el mecanismo seleccionado a cada muestra
    for i in range(n_samples):
        # Obtener el valor de muestra t_i
        t_i = t_i_tensor[i]

        # Aplicar el mecanismo elegido con epsilon ajustado
        if mechanism == 'duchi':
            x_i = duchi_mechanism(t_i.unsqueeze(0), epsilon_adjusted)
        elif mechanism == 'piecewise':
            x_i = piecewise_mechanism(t_i.unsqueeze(0), epsilon_adjusted)
        else:
            raise ValueError(f"Mecanismo desconocido: {mechanism}")

        # Escalar y asignar el valor a t_i_star
        t_i_star[i] = C * x_i.item()

    # Asegurarse de que los valores de t_i_star están dentro del rango [-C, C]
    t_i_star = torch.clamp(t_i_star, -C, C)

    return t_i_star

def apply_method(data_vector, method, epsilon):
    """
    Aplica el mecanismo de privacidad diferencial seleccionado al vector o matriz de datos,
    incluyendo la normalización y desnormalización de los datos.

    Parámetros:
    - data_vector (array-like o torch.Tensor): Vector o matriz de entrada con los datos originales.
    - method (str): Método de privatización a utilizar ('duchi', 'piecewise', 'laplace', 'multidimensional_duchi', 'multidimensional').
    - epsilon (float): Presupuesto de privacidad.

    Devuelve:
    - privatized_data (torch.Tensor): Datos privatizados en la escala original.
    """
    # Convertir data_vector a tensor de PyTorch si no lo es y asegurarse de que es float
    if not isinstance(data_vector, torch.Tensor):
        data_vector = torch.tensor(data_vector, dtype=torch.float32)
    else:
        data_vector = data_vector.clone().detach().float()

    # Si es unidimensional, convertirlo a 2D para unificar el procesamiento
    if data_vector.ndim == 1:
        data_vector = data_vector.unsqueeze(1)  # Convertir a matriz Nx1

    n_samples, n_features = data_vector.shape

    # Normalizar cada columna al rango [-1, 1]
    normalized_data = []
    min_vals = []
    max_vals = []
    for i in range(n_features):
        col = data_vector[:, i]
        normalized_col, min_val, max_val = normalize_to_range(col)
        normalized_data.append(normalized_col)
        min_vals.append(min_val)
        max_vals.append(max_val)
    # Crear matriz normalizada
    normalized_data = torch.stack(normalized_data, dim=1)

    # Aplicar el mecanismo según el método seleccionado
    if method in ['duchi', 'piecewise', 'laplace']:
        # Aplicar el mecanismo a cada columna individualmente
        transformed_cols = []
        for i in range(n_features):
            col = normalized_data[:, i]
            if method == 'duchi':
                transformed_col = duchi_mechanism(col, epsilon)
            elif method == 'piecewise':
                transformed_col = piecewise_mechanism(col, epsilon)
            elif method == 'laplace':
                transformed_col = laplace_mechanism(col, epsilon)
            transformed_cols.append(transformed_col)
        # Combinar las columnas transformadas
        transformed_data = torch.stack(transformed_cols, dim=1)
    elif method == 'multidimensional_duchi':
        # Aplicar el mecanismo multidimensional de Duchi
        transformed_data = multidimensional_duchi_mechanism(normalized_data, epsilon)
    elif method == 'multidimensional':
        # Aplicar nuestro mecanismo multidimensional
        transformed_data = multidimensional_mechanism(normalized_data, epsilon, mechanism='piecewise')
    else:
        raise ValueError(f"Método desconocido: {method}")

    # Desnormalizar cada columna al rango original
    privatized_cols = []
    for i in range(n_features):
        transformed_col = transformed_data[:, i]
        privatized_col = denormalize_from_range(
            transformed_col,
            min_val=min_vals[i],
            max_val=max_vals[i]
        )
        privatized_cols.append(privatized_col)
    # Crear matriz de datos privatizados
    privatized_data = torch.stack(privatized_cols, dim=1)

    # Si el data_vector original era unidimensional, devolver un vector unidimensional
    if privatized_data.shape[1] == 1:
        privatized_data = privatized_data.squeeze(1)

    return privatized_data