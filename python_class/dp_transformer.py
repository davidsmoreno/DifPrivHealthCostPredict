import numpy as np
import pandas as pd
from dp_mechanisms.quantitative import duchi_mechanism, piecewise_mechanism
from dp_mechanisms.categorical import direct_encoding, optimized_unary_encoding, rappor
from dp_plotting import plot_mean_vs_privatized, plot_histograms

class DifferentialPrivacyTransformer:
    def __init__(self, df, epsilon=1.0, quantitative_vars=None, qualitative_vars=None):
        """
        Inicializa el Transformador de Privacidad Diferencial.

        Parámetros:
        df (pd.DataFrame): DataFrame de entrada.
        epsilon (float): Presupuesto de privacidad. Por defecto es 1.0.
        quantitative_vars (list de str): Lista de variables cuantitativas.
        qualitative_vars (list de str): Lista de variables cualitativas.
        """
        self.df = df.copy()
        self.epsilon = epsilon
        self.quantitative_vars = quantitative_vars
        self.qualitative_vars = qualitative_vars
        self.df_transformed = df.copy()

        # Identificar variables si no se proporcionan
        if self.quantitative_vars is None and self.qualitative_vars is None:
            self.quantitative_vars = df.select_dtypes(include=np.number).columns.tolist()
            self.qualitative_vars = df.select_dtypes(exclude=np.number).columns.tolist()
        elif self.quantitative_vars is None:
            self.quantitative_vars = [col for col in df.columns if col not in self.qualitative_vars and pd.api.types.is_numeric_dtype(df[col])]
        elif self.qualitative_vars is None:
            self.qualitative_vars = [col for col in df.columns if col not in self.quantitative_vars and not pd.api.types.is_numeric_dtype(df[col])]

    def fit_quantitative(self, quantitative_vars=None, method='duchi'):
        """
        Aplica mecanismos de privacidad diferencial a variables cuantitativas.

        Parámetros:
        quantitative_vars (list de str): Lista de variables cuantitativas a transformar.
        method (str): Método a utilizar ('duchi' o 'piecewise').

        Devuelve:
        pd.DataFrame: DataFrame con variables cuantitativas transformadas.
        """
        # Usar variables proporcionadas o las de la clase
        if quantitative_vars is not None:
            self.quantitative_vars = quantitative_vars

        # Verificar que hay variables cuantitativas
        if not self.quantitative_vars:
            print("No hay variables cuantitativas para transformar.")
            return self.df_transformed

        # Aplicar el método seleccionado a cada variable
        for col in self.quantitative_vars:
            col_values = self.df[col].values.astype(float)
            col_min = np.min(col_values)
            col_max = np.max(col_values)
            if col_min == col_max:
                normalized_col = np.zeros_like(col_values)
            else:
                normalized_col = 2 * (col_values - col_min) / (col_max - col_min) - 1

            if method == 'duchi':
                transformed_col = duchi_mechanism(normalized_col, self.epsilon)
            elif method == 'piecewise':
                transformed_col = piecewise_mechanism(normalized_col, self.epsilon)
            else:
                raise ValueError(f"Método desconocido: {method}")

            denormalized_col = (transformed_col + 1) * (col_max - col_min) / 2 + col_min
            self.df_transformed[col] = denormalized_col

        return self.df_transformed

    def fit_qualitative(self, qualitative_vars=None, method='direct_encoding'):
        """
        Aplica mecanismos de privacidad diferencial a variables cualitativas.

        Parámetros:
        qualitative_vars (list de str): Lista de variables cualitativas a transformar.
        method (str): Método a utilizar ('direct_encoding', 'oue', 'rappor').

        Devuelve:
        pd.DataFrame: DataFrame con variables cualitativas transformadas.
        """
        # Usar variables proporcionadas o las de la clase
        if qualitative_vars is not None:
            self.qualitative_vars = qualitative_vars

        # Verificar que hay variables cualitativas
        if not self.qualitative_vars:
            print("No hay variables cualitativas para transformar.")
            return self.df_transformed

        # Aplicar el método seleccionado a cada variable
        for col in self.qualitative_vars:
            col_data = self.df[col].values
            if method == 'direct_encoding':
                transformed_col = direct_encoding(col_data, self.epsilon)
            elif method == 'oue':
                transformed_col = optimized_unary_encoding(col_data, self.epsilon)
            elif method == 'rappor':
                transformed_col = rappor(col_data, self.epsilon)
            else:
                raise ValueError(f"Método desconocido: {method}")
            self.df_transformed[col] = transformed_col

        return self.df_transformed

    def calculate_utility_metrics(self, variables=None):
        """
        Calcula métricas de utilidad para evaluar el impacto de la privatización.

        Parámetros:
        variables (list de str): Lista de variables cuantitativas para calcular métricas.

        Devuelve:
        pd.DataFrame: DataFrame con las métricas de utilidad.
        """
        if variables is None:
            variables = self.quantitative_vars

        metrics = []
        for variable in variables:
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

    def plot_mean_vs_privatized(self, variables=None, method='duchi', epsilons=None, **kwargs):
        """
        Grafica la media de las variables originales vs las privatizadas.

        Parámetros:
        variables (list de str): Variables a graficar.
        method (str): Método utilizado ('duchi' o 'piecewise').
        epsilons (list de float): Valores de epsilon a utilizar.
        **kwargs: Argumentos adicionales para personalizar los gráficos.

        Devuelve:
        None
        """
        if variables is None:
            variables = self.quantitative_vars

        plot_mean_vs_privatized(
            df_original=self.df,
            variables=variables,
            method=method,
            epsilons=epsilons,
            **kwargs
        )

    def plot_histograms(self, variables=None, method='duchi', epsilons=None, bins=30, **kwargs):
        """
        Genera histogramas de las variables cuantitativas.

        Parámetros:
        variables (list de str): Variables a graficar.
        method (str): Método utilizado ('duchi' o 'piecewise').
        epsilons (list de float): Valores de epsilon a utilizar.
        bins (int): Número de bins para los histogramas.
        **kwargs: Argumentos adicionales para personalizar los gráficos.

        Devuelve:
        None
        """
        if variables is None:
            variables = self.quantitative_vars

        plot_histograms(
            df_original=self.df,
            variables=variables,
            method=method,
            epsilons=epsilons,
            bins=bins,
            **kwargs
        )