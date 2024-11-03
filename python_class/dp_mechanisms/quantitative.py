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

def denormalize_from_range(normalized_vector, min_val, max_val, min_range=-1.0, max_range=1.0):
    """
    Desnormaliza un vector desde un rango especificado de vuelta a su rango original.

    Parámetros:
    - normalized_vector (torch.Tensor): El vector normalizado.
    - min_val (float): El valor mínimo del vector original.
    - max_val (float): El valor máximo del vector original.
    - min_range (float): El valor mínimo del rango normalizado. Por defecto es -1.0.
    - max_range (float): El valor máximo del rango normalizado. Por defecto es 1.0.

    Devuelve:
    - denormalized (torch.Tensor): El vector desnormalizado.
    """
    # Asegurar que normalized_vector es un tensor de tipo float32
    if not isinstance(normalized_vector, torch.Tensor):
        normalized_vector = torch.tensor(normalized_vector, dtype=torch.float32)
    else:
        normalized_vector = normalized_vector.clone().detach().float()

    # Escalar de vuelta a [0, 1]
    normalized_vector = (normalized_vector - min_range) / (max_range - min_range)
    # Escalar al rango original [min_val, max_val]
    denormalized = normalized_vector * (max_val - min_val) + min_val
    return denormalized

def duchi_mechanism(t_i_vector, epsilon):
    """
    Aplica el mecanismo de Duchi et al. para privacidad diferencial a un vector numérico.

    Parámetros:
    - t_i_vector (torch.Tensor o array-like): Vector de entrada de números en el rango [-1, 1].
    - epsilon (float): Presupuesto de privacidad (debe ser positivo).

    Devuelve:
    - t_i_star (torch.Tensor): Vector transformado con privacidad diferencial.
    """
    if epsilon <= 0:
        raise ValueError("Epsilon debe ser un valor positivo.")

    # Convertir el vector a tensor si no lo es
    if not isinstance(t_i_vector, torch.Tensor):
        t_i_tensor = torch.tensor(t_i_vector, dtype=torch.float32)
    else:
        t_i_tensor = t_i_vector.clone().detach().float()

    # Asegurar que los valores están en el rango [-1, 1]
    t_i_tensor = torch.clamp(t_i_tensor, -1.0, 1.0)

    # Calcular e^epsilon y expresiones recurrentes
    e_epsilon = torch.exp(torch.tensor(epsilon, dtype=torch.float32))
    e_epsilon_plus_1 = e_epsilon + 1.0
    e_epsilon_minus_1 = e_epsilon - 1.0

    # Calcular la probabilidad p para la respuesta aleatoria
    prob = 0.5 * (t_i_tensor + 1.0) * (e_epsilon / e_epsilon_plus_1) + \
           (0.5 * (1.0 - t_i_tensor)) * (1.0 / e_epsilon_plus_1)

    # Generar valores aleatorios según la distribución Bernoulli
    u = torch.bernoulli(prob)

    # Calcular el valor privatizado
    t_i_star = ((2 * u) - 1.0) * (e_epsilon_plus_1 / e_epsilon_minus_1)

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

def piecewise_mechanism(t_i_vector, epsilon):
    """
    Aplica el Mecanismo por Tramos para datos numéricos unidimensionales.

    Parámetros:
    - t_i_vector (torch.Tensor o array-like): Vector de entrada de números en el rango [-1, 1].
    - epsilon (float): Presupuesto de privacidad (debe ser positivo).

    Devuelve:
    - t_i_star (torch.Tensor): Vector transformado dentro del rango [-C, C].
    """
    if epsilon <= 0:
        raise ValueError("Epsilon debe ser un valor positivo.")

    # Convertir el vector a tensor si no lo es
    if not isinstance(t_i_vector, torch.Tensor):
        t_i_tensor = torch.tensor(t_i_vector, dtype=torch.float32)
    else:
        t_i_tensor = t_i_vector.clone().detach().float()

    # Asegurar que los valores están en el rango [-1, 1]
    t_i_tensor = torch.clamp(t_i_tensor, -1.0, 1.0)

    # Calcular e^(epsilon/2) y C
    e_epsilon_2 = torch.exp(torch.tensor(epsilon / 2.0))
    C = (e_epsilon_2 + 1.0) / (e_epsilon_2 - 1.0)

    # Calcular l(t_i) y r(t_i)
    l_t = ((C + 1.0) / 2.0) * t_i_tensor - (C - 1.0) / 2.0
    r_t = l_t + C - 1.0

    # Generar valores aleatorios para la decisión
    x = torch.rand_like(t_i_tensor)
    threshold = e_epsilon_2 / (e_epsilon_2 + 1.0)
    rand_uniform = torch.rand_like(t_i_tensor)

    # Inicializar el tensor para los valores privatizados
    t_i_star = torch.empty_like(t_i_tensor)

    # Máscara para x < threshold
    mask1 = x < threshold
    low1 = l_t
    high1 = r_t
    # Asegurar que high1 > low1
    high1 = torch.where(high1 <= low1, low1 + 1e-6, high1)
    t_i_star[mask1] = low1[mask1] + (high1[mask1] - low1[mask1]) * rand_uniform[mask1]

    # Máscara para x >= threshold
    mask2 = ~mask1
    interval_choice = torch.rand_like(t_i_tensor) < 0.5

    # Intervalo [-C, l(t_i)]
    mask2_interval1 = mask2 & interval_choice
    low2_interval1 = -C
    high2_interval1 = l_t[mask2_interval1]
    high2_interval1 = torch.where(high2_interval1 <= low2_interval1,
                                  low2_interval1 + 1e-6,
                                  high2_interval1)
    t_i_star[mask2_interval1] = low2_interval1 + (high2_interval1 - low2_interval1) * rand_uniform[mask2_interval1]

    # Intervalo [r(t_i), C]
    mask2_interval2 = mask2 & (~interval_choice)
    low2_interval2 = r_t[mask2_interval2]
    high2_interval2 = C
    high2_interval2 = torch.where(high2_interval2 <= low2_interval2,
                                  low2_interval2 + 1e-6,
                                  high2_interval2)
    t_i_star[mask2_interval2] = low2_interval2 + (high2_interval2 - low2_interval2) * rand_uniform[mask2_interval2]

    return t_i_star

def multidimensional_duchi_mechanism(t_i_tensor, epsilon, B=1.0):
    """
    Aplica el mecanismo de Duchi et al. para datos numéricos multidimensionales.

    Parámetros:
    - t_i_tensor (torch.Tensor): Tensor de entrada de forma (n_samples, d), con valores en el rango [-1, 1].
    - epsilon (float): Presupuesto de privacidad (debe ser positivo).
    - B (float): Valor absoluto máximo para las componentes de salida (debe ser positivo, por defecto 1.0).

    Devuelve:
    - t_i_star (torch.Tensor): Tensor transformado con privacidad diferencial, de forma (n_samples, d), con valores en {-B, B}^d.
    """
    if epsilon <= 0:
        raise ValueError("Epsilon debe ser un valor positivo.")
    if B <= 0:
        raise ValueError("B debe ser un valor positivo.")

    # Asegurar que los valores están en el rango [-1, 1]
    t_i_tensor = torch.clamp(t_i_tensor, -1.0, 1.0)

    n_samples, d = t_i_tensor.shape

    # Paso 1: Generar un vector aleatorio v ∈ {−1, 1}^d para cada muestra
    prob_v = 0.5 * (1.0 + t_i_tensor)  # Probabilidades para v[j] = 1, forma (n_samples, d)
    random_vals_v = torch.rand(n_samples, d)
    v = torch.where(random_vals_v < prob_v, torch.ones_like(prob_v), -torch.ones_like(prob_v))

    # Paso 3: Generar una variable Bernoulli u que es 1 con probabilidad e^epsilon / (e^epsilon + 1)
    e_epsilon = torch.exp(torch.tensor(epsilon, dtype=torch.float32))
    p_u = e_epsilon / (e_epsilon + 1.0)
    u = torch.bernoulli(torch.full((n_samples,), p_u))

    # Paso 4: Generar t_i_star ∈ {−B, B}^d uniformemente al azar para cada muestra
    random_vals_t = torch.rand(n_samples, d)
    t_i_star = torch.where(random_vals_t < 0.5, -B * torch.ones_like(random_vals_t), B * torch.ones_like(random_vals_t))

    # Calcular el producto punto s = t_i_star · v para cada muestra
    s = torch.sum(t_i_star * v, dim=1)  # Forma (n_samples,)

    # Paso 5: Asegurar que t_i_star pertenece a T+ o T− según el valor de u
    # Crear una máscara para las muestras que necesitan invertir t_i_star
    condition = ((u == 1.0) & (s < 0.0)) | ((u == 0.0) & (s > 0.0))
    # Invertir t_i_star para esas muestras
    t_i_star[condition] = -t_i_star[condition]

    return t_i_star

def multidimensional_mechanism(t_i_tensor, epsilon, mechanism='piecewise', C=1.0):
    """
    Aplica un mecanismo de privacidad diferencial para múltiples atributos numéricos.

    Parámetros:
    - t_i_tensor (torch.Tensor o array-like): Tensor de entrada de forma (n_samples, d), con valores en el rango [-1, 1].
    - epsilon (float): Presupuesto de privacidad (debe ser positivo).
    - mechanism (str): Mecanismo a utilizar ('piecewise', 'laplace' o 'duchi').
    - C (float): Constante para escalar los valores de salida (debe ser positivo, por defecto 1.0).

    Devuelve:
    - t_i_star (torch.Tensor): Tensor transformado con privacidad diferencial, de forma (n_samples, d), con valores en [-C·d, C·d]^d.
    """
    if epsilon <= 0:
        raise ValueError("Epsilon debe ser un valor positivo.")
    if C <= 0:
        raise ValueError("C debe ser un valor positivo.")

    # Convertir el vector a tensor si no lo es
    if not isinstance(t_i_tensor, torch.Tensor):
        t_i_tensor = torch.tensor(t_i_tensor, dtype=torch.float32)
    else:
        t_i_tensor = t_i_tensor.clone().detach().float()

    # Asegurar que los valores están en el rango [-1, 1]
    t_i_tensor = torch.clamp(t_i_tensor, -1.0, 1.0)
    n_samples, d = t_i_tensor.shape  # Dimensión del tensor

    # Paso 1: Inicializar t_i_star como matriz de ceros
    t_i_star = torch.zeros((n_samples, d), dtype=torch.float32)

    # Paso 2: Calcular k = max(1, min(d, floor(epsilon / 2.5)))
    k = max(1, min(d, int(epsilon // 2.5)))

    # Paso 3: Para cada muestra, seleccionar k índices aleatoriamente sin reemplazo
    for i in range(n_samples):
        indices = torch.randperm(d)[:k]

        # Paso 4 y 5: Para cada índice seleccionado, aplicar el mecanismo y asignar el valor
        for idx in indices:
            idx = idx.item()  # Convertir el índice a entero de Python

            # Obtener el valor t_i[Aj]
            t_i_j = t_i_tensor[i, idx]

            # Ajustar epsilon para este componente
            epsilon_adjusted = epsilon / k

            # Aplicar el mecanismo seleccionado con epsilon ajustado
            if mechanism == 'duchi':
                x_i_j = duchi_mechanism(t_i_j.unsqueeze(0), epsilon_adjusted)
            elif mechanism == 'piecewise':
                x_i_j = piecewise_mechanism(t_i_j.unsqueeze(0), epsilon_adjusted)
            elif mechanism == 'laplace':
                x_i_j = laplace_mechanism(t_i_j.unsqueeze(0), epsilon_adjusted)
            else:
                raise ValueError(f"Mecanismo desconocido: {mechanism}")

            # Paso 6: Escalar y asignar el valor a t_i_star[Aj]
            t_i_star[i, idx] = (d / k) * x_i_j.item()

    # Asegurar que los valores de t_i_star están en el rango [-C·d, C·d]
    t_i_star = torch.clamp(t_i_star, -C * d, C * d)

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