import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_mean_vs_privatized(df_original, variables, method, epsilons=None, **kwargs):
    """
    Grafica la media de las variables originales vs las privatizadas para diferentes valores de epsilon.

    Parámetros:
    df_original (pd.DataFrame): DataFrame original.
    variables (list de str): Lista de variables cuantitativas.
    method (str): Método utilizado ('duchi' o 'piecewise').
    epsilons (list de float): Valores de epsilon a graficar.
    **kwargs: Argumentos adicionales para personalizar los gráficos.

    Devuelve:
    None
    """
    if epsilons is None:
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    for variable in variables:
        original_mean = df_original[variable].mean()
        privatized_means = []

        col_values = df_original[variable].values.astype(float)
        col_min = np.min(col_values)
        col_max = np.max(col_values)
        if col_min == col_max:
            normalized_col = np.zeros_like(col_values)
        else:
            normalized_col = 2 * (col_values - col_min) / (col_max - col_min) - 1

        for eps in epsilons:
            if method == 'duchi':
                from dp_mechanisms.quantitative import duchi_mechanism
                transformed_col = duchi_mechanism(normalized_col, eps)
            elif method == 'piecewise':
                from dp_mechanisms.quantitative import piecewise_mechanism
                transformed_col = piecewise_mechanism(normalized_col, eps)
            else:
                raise ValueError(f"Método desconocido: {method}")
            denormalized_col = (transformed_col + 1) * (col_max - col_min) / 2 + col_min
            privatized_mean = denormalized_col.mean()
            privatized_means.append(privatized_mean)

        sns.set(style="whitegrid")
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        sns.lineplot(x=epsilons, y=[original_mean]*len(epsilons), label='Media Original', linestyle='--')
        sns.lineplot(x=epsilons, y=privatized_means, marker='o', label='Media Privatizada')
        plt.xlabel('Epsilon', fontsize=kwargs.get('fontsize', 12))
        plt.ylabel('Valor Medio', fontsize=kwargs.get('fontsize', 12))
        plt.title(f'Media Original vs. Privatizada para "{variable}"', fontsize=kwargs.get('fontsize', 14))
        plt.legend()
        plt.show()

def plot_histograms(df_original, variables, method, epsilons=None, bins=30, **kwargs):
    """
    Genera histogramas de las variables cuantitativas para diferentes valores de epsilon.

    Parámetros:
    df_original (pd.DataFrame): DataFrame original.
    variables (list de str): Lista de variables cuantitativas.
    method (str): Método utilizado ('duchi' o 'piecewise').
    epsilons (list de float): Valores de epsilon a graficar.
    bins (int): Número de bins para los histogramas.
    **kwargs: Argumentos adicionales para personalizar los gráficos.

    Devuelve:
    None
    """
    if epsilons is None:
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    for variable in variables:
        col_values = df_original[variable].values.astype(float)
        col_min = np.min(col_values)
        col_max = np.max(col_values)
        if col_min == col_max:
            normalized_col = np.zeros_like(col_values)
        else:
            normalized_col = 2 * (col_values - col_min) / (col_max - col_min) - 1

        plt.figure(figsize=kwargs.get('figsize', (15, 8)))
        plt.suptitle(f'Histogramas para "{variable}" con diferentes valores de epsilon', fontsize=kwargs.get('fontsize', 16))

        for idx, eps in enumerate(epsilons):
            if method == 'duchi':
                from dp_mechanisms.quantitative import duchi_mechanism
                transformed_col = duchi_mechanism(normalized_col, eps)
            elif method == 'piecewise':
                from dp_mechanisms.quantitative import piecewise_mechanism
                transformed_col = piecewise_mechanism(normalized_col, eps)
            else:
                raise ValueError(f"Método desconocido: {method}")
            denormalized_col = (transformed_col + 1) * (col_max - col_min) / 2 + col_min

            plt.subplot(2, 3, idx + 1)
            sns.histplot(denormalized_col, kde=True, bins=bins, color='skyblue', **kwargs)
            plt.title(f'Epsilon = {eps}', fontsize=kwargs.get('fontsize', 12))
            plt.xlabel(variable, fontsize=kwargs.get('fontsize', 10))
            plt.ylabel('Frecuencia', fontsize=kwargs.get('fontsize', 10))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()