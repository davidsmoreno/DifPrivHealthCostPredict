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
    Denormalize a vector from a specified range back to its original range.

    Parameters:
    normalized_vector (torch.Tensor): The normalized vector.
    min_val (float): The minimum value of the original vector.
    max_val (float): The maximum value of the original vector.
    min_range (float): The minimum value of the normalized range. Default is -1.
    max_range (float): The maximum value of the normalized range. Default is 1.

    Returns:
    torch.Tensor: The denormalized vector.
    """
    # Scale back to [0, 1]
    normalized_vector = (normalized_vector - min_range) / (max_range - min_range)
    # Scale to the original range [min_val, max_val]
    denormalized = normalized_vector * (max_val - min_val) + min_val
    return denormalized


def duchi_mechanism(t_i_vector, epsilon):
    """
    Applies the Duchi et al. mechanism for differential privacy to a numerical vector.

    Parameters:
    - t_i_vector (torch.Tensor or array-like): Input vector of numbers, assumed to be in the range [-1, 1].
    - epsilon (float): Privacy budget (must be positive).

    Returns:
    - t_i_star (torch.Tensor): Transformed vector with differential privacy.
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be a positive value.")

    # Convert the vector to a double-precision tensor if it’s not already
    if not isinstance(t_i_vector, torch.Tensor):
        t_i_tensor = torch.tensor(t_i_vector, dtype=torch.float64)
    else:
        t_i_tensor = t_i_vector.clone().detach().double()

    # Compute tanh(epsilon / 2) for privacy scaling
    epsilon_half = torch.tensor(epsilon / 2.0, dtype=torch.float64)
    tanh_epsilon_half = torch.tanh(epsilon_half)

    # Calculate probability p(x) for Bernoulli sampling
    prob = 0.5 * (1.0 + t_i_tensor * tanh_epsilon_half)

    # Ensure probabilities are within [0, 1] to avoid sampling issues
    prob = torch.clamp(prob, 0.0, 1.0)

    # Sample Bernoulli variable based on probability
    u = torch.bernoulli(prob)

    # Calculate the scaling factor
    w = 1.0 / tanh_epsilon_half

    # Apply the privatized transformation
    t_i_star = (2 * u - 1.0) * w

    return t_i_star



def piecewise_mechanism(t_i_vector, epsilon):
    """
    Apply the Piecewise Mechanism for one-dimensional numerical data.

    Parameters:
    t_i_vector (torch.Tensor or np.array): Input vector of numbers within the range [-1, 1].
    epsilon (float): Privacy budget.

    Returns:
    torch.Tensor: Transformed vector within the range [-C, C].
    """
    # Ensure t_i_vector is a torch.Tensor
    if not isinstance(t_i_vector, torch.Tensor):
        t_i_tensor = torch.from_numpy(np.array(t_i_vector)).float()
    else:
        t_i_tensor = t_i_vector.float()

    # Ensure values are within the range [-1, 1]
    t_i_tensor = torch.clamp(t_i_tensor, -1.0, 1.0)

    # Calculate e^(epsilon/2) and C
    e_epsilon_2 = torch.exp(torch.tensor(epsilon / 2.0))
    C = (e_epsilon_2 + 1.0) / (e_epsilon_2 - 1.0)

    # Calculate l(t_i) and r(t_i)
    l = ((C + 1.0) / 2.0) * t_i_tensor - (C - 1.0) / 2.0
    r = l + C - 1.0

    # Generate random values for the decision
    x = torch.rand_like(t_i_tensor)
    threshold = e_epsilon_2 / (e_epsilon_2 + 1.0)
    rand_uniform = torch.rand_like(t_i_tensor)

    # Initialize the tensor for privatized values
    t_i_star = torch.empty_like(t_i_tensor)

    # Mask for x < threshold
    mask1 = x < threshold
    low1 = l
    high1 = r
    high1 = torch.where(high1 <= low1, low1 + 1e-5, high1)
    t_i_star[mask1] = low1[mask1] + (high1[mask1] - low1[mask1]) * rand_uniform[mask1]

    # Mask for x >= threshold
    mask2 = ~mask1
    interval_choice = torch.rand_like(t_i_tensor) < 0.5

    # Combined masks
    mask2_interval1 = mask2 & interval_choice  # Interval [-C, l(t_i)]
    mask2_interval2 = mask2 & ~interval_choice  # Interval [r(t_i), C]

    # Create 'low' and 'high' tensors for each interval
    low2 = torch.empty_like(t_i_tensor)
    high2 = torch.empty_like(t_i_tensor)

    # Interval [-C, l(t_i)]
    low2[mask2_interval1] = -C
    high2[mask2_interval1] = l[mask2_interval1]
    high2[mask2_interval1] = torch.where(high2[mask2_interval1] <= low2[mask2_interval1],
                                         low2[mask2_interval1] + 1e-5,
                                         high2[mask2_interval1])
    t_i_star[mask2_interval1] = low2[mask2_interval1] + (high2[mask2_interval1] - low2[mask2_interval1]) * rand_uniform[mask2_interval1]

    # Interval [r(t_i), C]
    low2[mask2_interval2] = r[mask2_interval2]
    high2[mask2_interval2] = C
    high2[mask2_interval2] = torch.where(high2[mask2_interval2] <= low2[mask2_interval2],
                                         low2[mask2_interval2] + 1e-5,
                                         high2[mask2_interval2])
    t_i_star[mask2_interval2] = low2[mask2_interval2] + (high2[mask2_interval2] - low2[mask2_interval2]) * rand_uniform[mask2_interval2]

    # Return the tensor
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
    Applies the Duchi et al. mechanism for differential privacy to a single multidimensional vector with optimizations.

    Parameters:
    - t_i_vector (torch.Tensor): Input vector of shape (d,), with values in the range [-1, 1].
    - epsilon (float): Privacy budget (must be positive).
    - B (float): Absolute maximum value for output components (default is 1.0, must be positive).
    - num_samples (int): Number of samples to approximate T+ and T− (reduces exhaustive search).

    Returns:
    - t_i_star (torch.Tensor): Transformed vector with differential privacy, shape (d,), values in {-B, B}^d.
    """
    # Validate epsilon and B
    if epsilon <= 0:
        raise ValueError("Epsilon must be a positive value.")
    if B <= 0:
        raise ValueError("B must be a positive value.")

    # Ensure values in t_i_vector are within the range [-1, 1]
    t_i_vector = torch.clamp(t_i_vector, -1.0, 1.0)
    d = t_i_vector.shape[0]  # Dimension of the vector

    # Step 1: Generate random vector v ∈ {−1, 1}^d based on probabilities from t_i_vector
    prob_v = 0.5 * (1.0 + t_i_vector)  # Probabilities for v[j] = 1
    random_vals_v = torch.rand(d)
    v = torch.where(random_vals_v < prob_v, torch.ones(d), -torch.ones(d))  # Vector v ∈ {−1, 1}^d

    # Step 2: Sample T+ and T− with random tuples in {-B, B}^d instead of generating all combinations
    T_plus = []
    T_minus = []

    for _ in range(num_samples):
        # Generate a random candidate in {-B, B}^d
        random_bits = torch.randint(0, 2, (d,))  # Randomly generate bits {0, 1} for each component
        t_candidate = torch.where(random_bits == 1, B, -B)  # Map bits to {-B, B}
        
        # Classify based on dot product with v
        if torch.dot(t_candidate, v) >= 0:
            T_plus.append(t_candidate)
        else:
            T_minus.append(t_candidate)

    # Convert T_plus and T_minus to tensors for efficient random selection
    T_plus = torch.stack(T_plus) if T_plus else None
    T_minus = torch.stack(T_minus) if T_minus else None

    # Step 3: Sample a Bernoulli variable u that is 1 with probability e^epsilon / (e^epsilon + 1)
    e_epsilon = torch.exp(torch.tensor(epsilon, dtype=torch.float32))
    prob_u = e_epsilon / (e_epsilon + 1.0)
    u = torch.bernoulli(torch.tensor(prob_u))

    # Step 4: Select t_i_star from T+ or T− based on the sampled value of u
    if u == 1 and T_plus is not None:
        # Select uniformly at random from T+
        t_i_star = T_plus[torch.randint(len(T_plus), (1,)).item()]
    elif T_minus is not None:
        # Select uniformly at random from T−
        t_i_star = T_minus[torch.randint(len(T_minus), (1,)).item()]
    else:
        # Fallback in case either T_plus or T_minus is empty due to sampling
        t_i_star = torch.full((d,), B if u == 1 else -B, dtype=torch.float32)

    return t_i_star


def multidimensional_mechanism(t_i_vector, epsilon, mechanism='piecewise', C=1.0):
    """
    Applies a differential privacy mechanism to a 1-dimensional vector.

    Parameters:
    - t_i_vector (torch.Tensor or array-like): Input vector of shape (n_samples,), with values in the range [-1, 1].
    - epsilon (float): Privacy budget (must be positive).
    - mechanism (str): Mechanism to use ('piecewise', 'laplace' or 'duchi').
    - C (float): Constant to scale the output values (must be positive, default is 1.0).

    Returns:
    - t_i_star (torch.Tensor): Transformed vector with differential privacy, of shape (n_samples,), with values in the range [-C, C].
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be a positive value.")
    if C <= 0:
        raise ValueError("C must be a positive value.")

    # Convert the input to a tensor if it's not already
    if not isinstance(t_i_vector, torch.Tensor):
        t_i_tensor = torch.tensor(t_i_vector, dtype=torch.float32)
    else:
        t_i_tensor = t_i_vector.clone().detach().float()

    # Ensure values are within the range [-1, 1]
    t_i_tensor = torch.clamp(t_i_tensor, -1.0, 1.0)
    n_samples = t_i_tensor.shape[0]  # Number of samples

    # Initialize t_i_star as a zero tensor of shape (n_samples,)
    t_i_star = torch.zeros(n_samples, dtype=torch.float32)

    # Adjust epsilon for each sample directly
    epsilon_adjusted = epsilon

    # Apply the selected mechanism to each sample
    for i in range(n_samples):
        # Get the sample value t_i
        t_i = t_i_tensor[i]

        # Apply the chosen mechanism with adjusted epsilon
        if mechanism == 'duchi':
            x_i = duchi_mechanism(t_i.unsqueeze(0), epsilon_adjusted)
        elif mechanism == 'piecewise':
            x_i = piecewise_mechanism(t_i.unsqueeze(0), epsilon_adjusted)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        # Scale and assign the value to t_i_star
        t_i_star[i] = C * x_i.item()

    # Ensure that t_i_star values are within the range [-C, C]
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