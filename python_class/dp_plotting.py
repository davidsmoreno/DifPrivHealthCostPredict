import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_mean_vs_privatized(df_original, variables, quantitative_method, epsilons=None, **kwargs):
    """
    Grafica la media de las variables originales vs las variables privatizadas para diferentes valores de epsilon.

    Parámetros:
    df_original (pd.DataFrame): DataFrame original.
    variables (list de str): Lista de nombres de variables cuantitativas.
    quantitative_method (str): Método utilizado para variables cuantitativas ('duchi' o 'piecewise').
    epsilons (list de float): Lista de valores de epsilon a graficar. Si es None, se utilizan valores comunes.
    **kwargs: Argumentos adicionales para personalizar los gráficos.

    Devuelve:
    None
    """
    # Valores de epsilon por defecto si no se proporcionan
    if epsilons is None:
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    for variable in variables:
        original_mean = df_original[variable].mean()
        privatized_means = []

        # Obtener los valores de la columna
        col_values = df_original[variable].values.astype(float)
        col_min = np.min(col_values)
        col_max = np.max(col_values)
        # Normalizar la columna al rango [-1, 1]
        if col_min == col_max:
            normalized_col = np.zeros_like(col_values)
        else:
            normalized_col = 2 * (col_values - col_min) / (col_max - col_min) - 1

        # Calcular la media privatizada para cada epsilon
        for eps in epsilons:
            if quantitative_method == 'duchi':
                from dp_mechanisms.quantitative import duchi_mechanism
                transformed_col = duchi_mechanism(normalized_col, eps)
            elif quantitative_method == 'piecewise':
                from dp_mechanisms.quantitative import piecewise_mechanism
                transformed_col = piecewise_mechanism(normalized_col, eps)
            else:
                raise ValueError(f"Método cuantitativo desconocido: {quantitative_method}")
            denormalized_col = (transformed_col + 1) * (col_max - col_min) / 2 + col_min
            privatized_mean = denormalized_col.mean()
            privatized_means.append(privatized_mean)

        # Crear la gráfica usando seaborn
        sns.set(style="whitegrid")
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        sns.lineplot(x=epsilons, y=[original_mean]*len(epsilons), label='Media Original', linestyle='--')
        sns.lineplot(x=epsilons, y=privatized_means, marker='o', label='Media Privatizada')
        plt.xlabel('Epsilon', fontsize=kwargs.get('fontsize', 12))
        plt.ylabel('Valor Medio', fontsize=kwargs.get('fontsize', 12))
        plt.title(f'Media Original vs. Privatizada para "{variable}"', fontsize=kwargs.get('fontsize', 14))
        plt.legend()
        plt.show()

def plot_histograms(df_original, variables, quantitative_method, epsilons=None, bins=30, **kwargs):
    """
    Genera histogramas de frecuencias para cada variable cuantitativa incluida, para cada uno de los valores de epsilon.

    Parámetros:
    df_original (pd.DataFrame): DataFrame original.
    variables (list de str): Lista de nombres de variables cuantitativas.
    quantitative_method (str): Método utilizado para variables cuantitativas ('duchi' o 'piecewise').
    epsilons (list de float): Lista de valores de epsilon a graficar. Si es None, se utilizan valores comunes.
    bins (int): Número de bins para el histograma.
    **kwargs: Argumentos adicionales para personalizar los gráficos.

    Devuelve:
    None
    """
    # Valores de epsilon por defecto si no se proporcionan
    if epsilons is None:
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    for variable in variables:
        # Obtener los valores de la columna
        col_values = df_original[variable].values.astype(float)
        col_min = np.min(col_values)
        col_max = np.max(col_values)
        # Normalizar la columna al rango [-1, 1]
        if col_min == col_max:
            normalized_col = np.zeros_like(col_values)
        else:
            normalized_col = 2 * (col_values - col_min) / (col_max - col_min) - 1

        # Crear una figura para la variable actual
        plt.figure(figsize=kwargs.get('figsize', (15, 8)))
        plt.suptitle(f'Histogramas para "{variable}" con diferentes valores de epsilon', fontsize=kwargs.get('fontsize', 16))

        # Iterar sobre los valores de epsilon
        for idx, eps in enumerate(epsilons):
            if quantitative_method == 'duchi':
                from dp_mechanisms.quantitative import duchi_mechanism
                transformed_col = duchi_mechanism(normalized_col, eps)
            elif quantitative_method == 'piecewise':
                from dp_mechanisms.quantitative import piecewise_mechanism
                transformed_col = piecewise_mechanism(normalized_col, eps)
            else:
                raise ValueError(f"Método cuantitativo desconocido: {quantitative_method}")
            denormalized_col = (transformed_col + 1) * (col_max - col_min) / 2 + col_min

            # Crear el histograma usando seaborn
            plt.subplot(2, 3, idx + 1)
            sns.histplot(denormalized_col, kde=True, bins=bins, color='skyblue', **kwargs)
            plt.title(f'Epsilon = {eps}', fontsize=kwargs.get('fontsize', 12))
            plt.xlabel(variable, fontsize=kwargs.get('fontsize', 10))
            plt.ylabel('Frecuencia', fontsize=kwargs.get('fontsize', 10))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()