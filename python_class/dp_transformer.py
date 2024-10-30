import numpy as np
import pandas as pd
from dp_mechanisms.quantitative import duchi_mechanism, piecewise_mechanism
from dp_mechanisms.categorical import direct_encoding, optimized_unary_encoding, rappor
from dp_plotting import plot_mean_vs_privatized, plot_histograms

class DifferentialPrivacyTransformer:
    def __init__(self, df, epsilon, 
                 quantitative_vars=None, 
                 qualitative_vars=None,
                 quantitative_method='duchi',
                 qualitative_method='direct_encoding'):
        """
        Inicializa el Transformador de Privacidad Diferencial.

        Parámetros:
        df (pd.DataFrame): DataFrame de entrada.
        epsilon (float): Presupuesto de privacidad.
        quantitative_vars (list de str): Lista de nombres de variables cuantitativas.
        qualitative_vars (list de str): Lista de nombres de variables cualitativas.
        quantitative_method (str): Método para variables cuantitativas ('duchi' o 'piecewise').
        qualitative_method (str): Método para variables cualitativas ('direct_encoding', 'oue', 'rappor').
        """
        # Guardar el DataFrame original
        self.df = df.copy()
        self.epsilon = epsilon
        self.quantitative_vars = quantitative_vars
        self.qualitative_vars = qualitative_vars
        self.quantitative_method = quantitative_method
        self.qualitative_method = qualitative_method

        # Identificar tipos de variables si no se proporcionan
        if self.quantitative_vars is None and self.qualitative_vars is None:
            self.quantitative_vars = df.select_dtypes(include=np.number).columns.tolist()
            self.qualitative_vars = df.select_dtypes(exclude=np.number).columns.tolist()
        elif self.quantitative_vars is None:
            self.quantitative_vars = [col for col in df.columns if col not in self.qualitative_vars and pd.api.types.is_numeric_dtype(df[col])]
        elif self.qualitative_vars is None:
            self.qualitative_vars = [col for col in df.columns if col not in self.quantitative_vars and not pd.api.types.is_numeric_dtype(df[col])]

    def fit_transform(self):
        """
        Aplica mecanismos de privacidad diferencial al DataFrame.

        Devuelve:
        pd.DataFrame: DataFrame transformado con privacidad diferencial aplicada.
        """
        df_transformed = self.df.copy()

        # Aplicar método cuantitativo
        for col in self.quantitative_vars:
            # Obtener los valores de la columna
            col_values = self.df[col].values.astype(float)
            col_min = np.min(col_values)
            col_max = np.max(col_values)
            # Normalizar la columna al rango [-1, 1]
            if col_min == col_max:
                normalized_col = np.zeros_like(col_values)
            else:
                normalized_col = 2 * (col_values - col_min) / (col_max - col_min) - 1

            # Aplicar el mecanismo seleccionado
            if self.quantitative_method == 'duchi':
                transformed_col = duchi_mechanism(normalized_col, self.epsilon)
            elif self.quantitative_method == 'piecewise':
                transformed_col = piecewise_mechanism(normalized_col, self.epsilon)
            else:
                raise ValueError(f"Método cuantitativo desconocido: {self.quantitative_method}")

            # Desnormalizar la columna para mantener la escala original
            denormalized_col = (transformed_col + 1) * (col_max - col_min) / 2 + col_min
            df_transformed[col] = denormalized_col

        # Aplicar método cualitativo
        for col in self.qualitative_vars:
            col_data = self.df[col].values
            if self.qualitative_method == 'direct_encoding':
                transformed_col = direct_encoding(col_data, self.epsilon)
            elif self.qualitative_method == 'oue':
                transformed_col = optimized_unary_encoding(col_data, self.epsilon)
            elif self.qualitative_method == 'rappor':
                transformed_col = rappor(col_data, self.epsilon)
            else:
                raise ValueError(f"Método cualitativo desconocido: {self.qualitative_method}")
            df_transformed[col] = transformed_col

        # Guardar el DataFrame transformado
        self.df_transformed = df_transformed
        return df_transformed

    def calculate_utility_metrics(self):
        """
        Calcula métricas de utilidad para evaluar el impacto de la privatización.

        Devuelve:
        pd.DataFrame: DataFrame con las métricas de utilidad para cada variable cuantitativa.
        """
        if not hasattr(self, 'df_transformed'):
            raise AttributeError("Debe ejecutar 'fit_transform' antes de calcular las métricas.")

        metrics = []
        for variable in self.quantitative_vars:
            original = self.df[variable].values.astype(float)
            privatized = self.df_transformed[variable].values.astype(float)
            mse = np.mean((original - privatized) ** 2)
            mae = np.mean(np.abs(original - privatized))
            correlation = np.corrcoef(original, privatized)[0, 1]
            metrics.append({
                'Variable': variable,
                'MSE': mse,
                'MAE': mae,
                'Correlación': correlation
            })
        return pd.DataFrame(metrics)

    def plot_mean_vs_privatized(self, epsilons=None, **kwargs):
        """
        Grafica la media de las variables originales vs las variables privatizadas para diferentes valores de epsilon.

        Parámetros:
        epsilons (list de float): Lista de valores de epsilon a graficar. Si es None, se utilizan valores comunes.
        **kwargs: Argumentos adicionales para personalizar los gráficos.

        Devuelve:
        None
        """
        # Llamar a la función de graficación desde dp_plotting
        plot_mean_vs_privatized(
            df_original=self.df,
            variables=self.quantitative_vars,
            quantitative_method=self.quantitative_method,
            epsilons=epsilons,
            **kwargs
        )

    def plot_histograms(self, epsilons=None, bins=30, **kwargs):
        """
        Genera histogramas de frecuencias para cada variable cuantitativa incluida, para cada uno de los valores de epsilon.

        Parámetros:
        epsilons (list de float): Lista de valores de epsilon a graficar. Si es None, se utilizan valores comunes.
        bins (int): Número de bins para el histograma.
        **kwargs: Argumentos adicionales para personalizar los gráficos.

        Devuelve:
        None
        """
        # Llamar a la función de graficación desde dp_plotting
        plot_histograms(
            df_original=self.df,
            variables=self.quantitative_vars,
            quantitative_method=self.quantitative_method,
            epsilons=epsilons,
            bins=bins,
            **kwargs
        )