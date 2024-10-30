import numpy as np
import pandas as pd
from dp_mechanisms.quantitative import (
    duchi_mechanism,
    piecewise_mechanism,
    normalize_to_range,
    denormalize_from_range
)
from dp_mechanisms.categorical import direct_encoding, optimized_unary_encoding, rappor
from dp_plotting import plot_mean_vs_privatized, plot_histograms
import torch

class DifferentialPrivacyTransformer:
    def __init__(self, df, epsilon_quantitative=1.0, epsilon_categorical=1.0, quantitative_vars=None, categorical_vars=None):
        """
        Inicializa el Transformador de Privacidad Diferencial.

        Parámetros:
        df (pd.DataFrame): DataFrame de entrada.
        epsilon_quantitative (float): Presupuesto de privacidad para variables cuantitativas. Por defecto es 1.0.
        epsilon_categorical (float): Presupuesto de privacidad para variables categóricas. Por defecto es 1.0.
        quantitative_vars (list de str): Lista de variables cuantitativas.
        categorical_vars (list de str): Lista de variables categóricas.
        """
        self.df = df.copy()
        self.epsilon_quantitative = epsilon_quantitative
        self.epsilon_categorical = epsilon_categorical
        self.quantitative_vars = quantitative_vars
        self.categorical_vars = categorical_vars
        self.df_transformed = df.copy()
        self.min_max_vals = {}  # Para almacenar valores mínimos y máximos para desnormalización

        # Identificar variables si no se proporcionan
        if self.quantitative_vars is None and self.categorical_vars is None:
            self.quantitative_vars = df.select_dtypes(include=np.number).columns.tolist()
            self.categorical_vars = df.select_dtypes(exclude=np.number).columns.tolist()
        elif self.quantitative_vars is None:
            self.quantitative_vars = [
                col for col in df.columns
                if col not in self.categorical_vars and pd.api.types.is_numeric_dtype(df[col])
            ]
        elif self.categorical_vars is None:
            self.categorical_vars = [
                col for col in df.columns
                if col not in self.quantitative_vars and not pd.api.types.is_numeric_dtype(df[col])
            ]

    def fit_quantitative(self, quantitative_vars=None, method='duchi', epsilon=None):
        """
        Aplica mecanismos de privacidad diferencial a variables cuantitativas.

        Parámetros:
        quantitative_vars (list de str): Lista de variables cuantitativas a transformar.
        method (str): Método a utilizar ('duchi' o 'piecewise').
        epsilon (float): Presupuesto de privacidad para variables cuantitativas. Si es None, se usa el de la clase.

        Devuelve:
        pd.DataFrame: DataFrame con variables cuantitativas transformadas.
        """
        # Usar variables proporcionadas o las de la clase
        if quantitative_vars is not None:
            self.quantitative_vars = quantitative_vars

        # Usar epsilon proporcionado o el de la clase
        if epsilon is not None:
            epsilon_q = epsilon
        else:
            epsilon_q = self.epsilon_quantitative

        # Verificar que hay variables cuantitativas
        if not self.quantitative_vars:
            print("No hay variables cuantitativas para transformar.")
            return self.df_transformed

        # Aplicar el método seleccionado a cada variable
        for col in self.quantitative_vars:
            col_values = self.df[col].values.astype(float)
            # Normalizar los datos y almacenar valores mínimos y máximos
            normalized_col, min_val, max_val = normalize_to_range(col_values)
            self.min_max_vals[col] = (min_val, max_val)

            # Aplicar el mecanismo seleccionado
            if method == 'duchi':
                transformed_col = duchi_mechanism(normalized_col.numpy(), epsilon_q)
            elif method == 'piecewise':
                transformed_col = piecewise_mechanism(normalized_col.numpy(), epsilon_q)
            else:
                raise ValueError(f"Método desconocido: {method}")

            # Desnormalizar los datos transformados
            denormalized_col = denormalize_from_range(
                torch.tensor(transformed_col),
                min_val,
                max_val
            ).numpy()

            self.df_transformed[col] = denormalized_col

        return self.df_transformed

    def fit_categorical(self, categorical_vars=None, method='direct_encoding', epsilon=None):
        """
        Aplica mecanismos de privacidad diferencial a variables categóricas.

        Parámetros:
        categorical_vars (list de str): Lista de variables categóricas a transformar.
        method (str): Método a utilizar ('direct_encoding', 'oue', 'rappor').
        epsilon (float): Presupuesto de privacidad para variables categóricas. Si es None, se usa el de la clase.

        Devuelve:
        pd.DataFrame: DataFrame con variables categóricas transformadas.
        """
        # Usar variables proporcionadas o las de la clase
        if categorical_vars is not None:
            self.categorical_vars = categorical_vars

        # Usar epsilon proporcionado o el de la clase
        if epsilon is not None:
            epsilon_c = epsilon
        else:
            epsilon_c = self.epsilon_categorical

        # Verificar que hay variables categóricas
        if not self.categorical_vars:
            print("No hay variables categóricas para transformar.")
            return self.df_transformed

        # Aplicar el método seleccionado a cada variable
        for col in self.categorical_vars:
            col_data = self.df[col].values
            if method == 'direct_encoding':
                transformed_col = direct_encoding(col_data, epsilon_c)
            elif method == 'oue':
                transformed_col = optimized_unary_encoding(col_data, epsilon_c)
            elif method == 'rappor':
                transformed_col = rappor(col_data, epsilon_c)
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