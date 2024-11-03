import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from dp_mechanisms.quantitative import (
    apply_method
)

def plot_mean_vs_privatized(df_original, variables, method, epsilons=None, **kwargs):
    """
    Grafica la media de las variables originales vs. las privatizadas para diferentes valores de epsilon,
    mostrando barras de error que representan el error cuadrático medio (RMSE) entre la variable original y
    la variable privatizada para cada epsilon.

    Parámetros:
    - df_original (pd.DataFrame): DataFrame original.
    - variables (list de str): Lista de variables cuantitativas.
    - method (str): Método utilizado ('duchi', 'piecewise', 'laplace', 'multidimensional_duchi', 'multidimensional').
    - epsilons (list de float): Valores de epsilon a graficar.
    - **kwargs: Argumentos adicionales para personalizar los gráficos.

    Devuelve:
    - None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch

    if epsilons is None:
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    # Obtener los valores originales de las variables seleccionadas y convertirlos a float
    original_values = df_original[variables].values.astype(float)
    # Calcular las medias originales
    original_means = original_values.mean(axis=0)
    privatized_means = []
    rmse_list = []

    for eps in epsilons:
        # Aplicar el mecanismo de privatización utilizando apply_method
        privatized_data = apply_method(original_values, method, eps)

        # Convertir a tensores si no lo son
        if not isinstance(privatized_data, torch.Tensor):
            privatized_data = torch.tensor(privatized_data, dtype=torch.float32)

        # Calcular las medias de los datos privatizados
        privatized_mean = privatized_data.mean(axis=0).numpy()
        privatized_means.append(privatized_mean)

        # Calcular el RMSE entre los datos originales y los privatizados
        rmse = torch.sqrt(torch.mean((torch.tensor(original_values) - privatized_data) ** 2)).item()
        rmse_list.append(rmse)

    # Convertir listas a arrays para facilitar el manejo
    privatized_means = np.array(privatized_means)
    rmse_list = np.array(rmse_list)

    # Manejar el caso cuando solo hay una variable
    if len(variables) == 1:
        # Convertir las matrices a arreglos unidimensionales
        privatized_means = privatized_means.flatten()
        original_mean = original_means[0]

        sns.set(style="whitegrid")
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        plt.plot(epsilons, [original_mean]*len(epsilons), label='Media Original', linestyle='--')
        plt.errorbar(epsilons, privatized_means, yerr=rmse_list, fmt='-o', markersize=5, label='Media Privatizada')
        plt.xlabel('Epsilon', fontsize=kwargs.get('fontsize', 12))
        plt.ylabel('Valor Medio', fontsize=kwargs.get('fontsize', 12))
        plt.title(f'Media Original vs. Privatizada para "{variables[0]}" (Método = {method})', fontsize=kwargs.get('fontsize', 14))
        plt.legend()
        plt.show()
    else:
        # Graficar para cada variable
        for idx, variable in enumerate(variables):
            sns.set(style="whitegrid")
            plt.figure(figsize=kwargs.get('figsize', (10, 6)))
            plt.plot(epsilons, [original_means[idx]]*len(epsilons), label='Media Original', linestyle='--')
            plt.errorbar(epsilons, privatized_means[:, idx], yerr=rmse_list, fmt='-o', markersize=5, label='Media Privatizada')
            plt.xlabel('Epsilon', fontsize=kwargs.get('fontsize', 12))
            plt.ylabel('Valor Medio', fontsize=kwargs.get('fontsize', 12))
            plt.title(f'Media Original vs. Privatizada para "{variable}" (Método = {method})', fontsize=kwargs.get('fontsize', 14))
            plt.legend()
            plt.show()

def plot_histograms(df_original, variables, method, epsilon=1.0, bins=30, **kwargs):
    """
    Genera histogramas comparativos de las variables originales y privatizadas para un valor de epsilon dado.

    Parámetros:
    - df_original (pd.DataFrame): DataFrame original.
    - variables (list de str): Lista de variables cuantitativas.
    - method (str): Método utilizado ('duchi', 'piecewise', 'laplace', 'multidimensional_duchi', 'multidimensional').
    - epsilon (float): Valor de epsilon para la privatización (por defecto 1.0).
    - bins (int): Número de bins para los histogramas.
    - **kwargs: Argumentos adicionales para personalizar los gráficos.

    Devuelve:
    - None
    """
    # Obtener los valores de las variables seleccionadas
    original_values = df_original[variables].values.astype(float)

    # Asegurar que original_values es un array 2D
    if original_values.ndim == 1:
        original_values = original_values.reshape(-1, 1)

    # Aplicar el mecanismo de privatización utilizando apply_method
    privatized_data = apply_method(original_values, method, epsilon)

    # Convertir los datos a NumPy arrays para graficar
    if isinstance(privatized_data, torch.Tensor):
        privatized_data = privatized_data.numpy()

    # Asegurar que privatized_data es un array 2D
    if privatized_data.ndim == 1:
        privatized_data = privatized_data.reshape(-1, 1)

    # Graficar histogramas para cada variable
    for idx, variable in enumerate(variables):
        original_col_np = original_values[:, idx]
        privatized_col_np = privatized_data[:, idx]

        sns.set(style="whitegrid")
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        sns.histplot(original_col_np, bins=bins, color='blue', label='Original', kde=True, stat='density', alpha=0.5)
        sns.histplot(privatized_col_np, bins=bins, color='red', label='Privatizada', kde=True, stat='density', alpha=0.5)
        plt.title(f'Histograma de "{variable}" (Epsilon = {epsilon}, Método = {method})', fontsize=kwargs.get('fontsize', 14))
        plt.xlabel(variable, fontsize=kwargs.get('fontsize', 12))
        plt.ylabel('Densidad', fontsize=kwargs.get('fontsize', 12))
        plt.legend()
        plt.show()