import numpy as np
import pandas as pd
import torch

# Importar los mecanismos y funciones desde dp_mechanisms.quantitative
from dp_mechanisms.quantitative import (
    apply_method
)
# Importar los mecanismos para datos categóricos
from dp_mechanisms.categorical import direct_encoding, optimized_unary_encoding, rappor

# Importar funciones de graficación
from dp_plotting import plot_mean_vs_privatized, plot_histograms

class DifferentialPrivacyTransformer:
    def __init__(self, df, epsilon_quantitative=1.0, epsilon_categorical=1.0, quantitative_vars=None, categorical_vars=None):
        """
        Inicializa el Transformador de Privacidad Diferencial.

        Parámetros:
        - df (pd.DataFrame): DataFrame de entrada.
        - epsilon_quantitative (float): Presupuesto de privacidad para variables cuantitativas (por defecto 1.0).
        - epsilon_categorical (float): Presupuesto de privacidad para variables categóricas (por defecto 1.0).
        - quantitative_vars (list de str): Lista de variables cuantitativas.
        - categorical_vars (list de str): Lista de variables categóricas.
        """
        # Copiar el DataFrame original para evitar modificarlo
        self.df = df.copy()
        self.epsilon_quantitative = epsilon_quantitative
        self.epsilon_categorical = epsilon_categorical
        self.quantitative_vars = quantitative_vars
        self.categorical_vars = categorical_vars
        self.df_transformed = df.copy()  # DataFrame para almacenar los datos transformados

        # Identificar variables cuantitativas y categóricas si no se proporcionan
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
        - quantitative_vars (list de str): Lista de variables cuantitativas a transformar.
        - method (str): Método a utilizar ('duchi', 'laplace', 'piecewise', 'multidimensional_duchi', 'multidimensional').
        - epsilon (float): Presupuesto de privacidad para variables cuantitativas. Si es None, se usa el de la clase.

        Devuelve:
        - pd.DataFrame: DataFrame con variables cuantitativas transformadas.
        """
        # Usar variables proporcionadas o las de la clase
        if quantitative_vars is not None:
            self.quantitative_vars = quantitative_vars

        # Usar epsilon proporcionado o el de la clase
        epsilon_q = epsilon if epsilon is not None else self.epsilon_quantitative

        # Verificar que hay variables cuantitativas
        if not self.quantitative_vars:
            print("No hay variables cuantitativas para transformar.")
            return self.df_transformed

        # Obtener los datos de las variables cuantitativas
        data_quantitative = self.df[self.quantitative_vars].values.astype(float)

        # Aplicar el mecanismo de privacidad diferencial utilizando apply_method
        privatized_data = apply_method(data_quantitative, method, epsilon_q)

        # Crear un DataFrame con los datos privatizados
        df_privatized = pd.DataFrame(privatized_data, columns=self.quantitative_vars, index=self.df.index)

        # Actualizar el DataFrame transformado con los datos privatizados
        self.df_transformed.update(df_privatized)

        return self.df_transformed

    def fit_categorical(self, categorical_vars=None, method='direct_encoding', epsilon=None):
        """
        Aplica mecanismos de privacidad diferencial a variables categóricas.

        Parámetros:
        - categorical_vars (list de str): Lista de variables categóricas a transformar.
        - method (str): Método a utilizar ('direct_encoding', 'oue', 'rappor').
        - epsilon (float): Presupuesto de privacidad para variables categóricas. Si es None, se usa el de la clase.

        Devuelve:
        - pd.DataFrame: DataFrame con variables categóricas transformadas.
        """
        # Usar variables proporcionadas o las de la clase
        if categorical_vars is not None:
            self.categorical_vars = categorical_vars

        # Usar epsilon proporcionado o el de la clase
        epsilon_c = epsilon if epsilon is not None else self.epsilon_categorical

        # Verificar que hay variables categóricas
        if not self.categorical_vars:
            print("No hay variables categóricas para transformar.")
            return self.df_transformed

        # Aplicar el método seleccionado a cada variable categórica
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
            # Actualizar el DataFrame transformado con los datos privatizados
            self.df_transformed[col] = transformed_col

        return self.df_transformed

    def calculate_utility_metrics(self, variables=None):
        """
        Calcula métricas de utilidad para evaluar el impacto de la privatización.

        Parámetros:
        - variables (list de str): Lista de variables cuantitativas para calcular métricas.

        Devuelve:
        - pd.DataFrame: DataFrame con las métricas de utilidad.
        """
        # Usar variables proporcionadas o las de la clase
        if variables is None:
            variables = self.quantitative_vars

        metrics = []
        for variable in variables:
            # Obtener los datos originales y privatizados
            original = self.df[variable].values.astype(float)
            privatized = self.df_transformed[variable].values.astype(float)

            # Convertir a tensores de PyTorch
            original_tensor = torch.tensor(original)
            privatized_tensor = torch.tensor(privatized)

            # Calcular métricas utilizando PyTorch
            mse = torch.mean((original_tensor - privatized_tensor) ** 2).item()
            mae = torch.mean(torch.abs(original_tensor - privatized_tensor)).item()
            # Calcular la correlación, manejando casos donde la desviación estándar es cero
            if torch.std(original_tensor) > 0 and torch.std(privatized_tensor) > 0:
                correlation = torch.corrcoef(torch.stack([original_tensor, privatized_tensor]))[0, 1].item()
            else:
                correlation = float('nan')  # Correlación indefinida si la desviación estándar es cero

            # Almacenar las métricas en una lista
            metrics.append({
                'Variable': variable,
                'MSE': mse,
                'MAE': mae,
                'Correlación': correlation
            })

        # Crear un DataFrame con las métricas
        metrics_df = pd.DataFrame(metrics)
        return metrics_df

    def plot_mean_vs_privatized(self, variables=None, method='duchi', epsilons=None, **kwargs):
        """
        Grafica la media de las variables originales vs las privatizadas.

        Parámetros:
        - variables (list de str): Variables a graficar.
        - method (str): Método utilizado ('duchi', 'laplace', 'piecewise', 'multidimensional_duchi', 'multidimensional').
        - epsilons (list de float): Valores de epsilon a utilizar.
        - **kwargs: Argumentos adicionales para personalizar los gráficos.

        Devuelve:
        - None
        """
        # Usar variables proporcionadas o las de la clase
        if variables is None:
            variables = self.quantitative_vars

        # Llamar a la función de graficación utilizando los datos originales
        plot_mean_vs_privatized(
            df_original=self.df,
            variables=variables,
            method=method,
            epsilons=epsilons,
            **kwargs
        )

    def plot_histograms(self, variables=None, method='duchi', epsilon=1.0, bins=30, **kwargs):
        """
        Genera histogramas comparativos de las variables originales y privatizadas.

        Parámetros:
        - variables (list de str): Variables a graficar.
        - method (str): Método utilizado ('duchi', 'laplace', 'piecewise', 'multidimensional_duchi', 'multidimensional').
        - epsilon (float): Valor de epsilon a utilizar.
        - bins (int): Número de bins para los histogramas.
        - **kwargs: Argumentos adicionales para personalizar los gráficos.

        Devuelve:
        - None
        """
        # Usar variables proporcionadas o las de la clase
        if variables is None:
            variables = self.quantitative_vars

        # Llamar a la función de graficación de histogramas comparativos
        plot_histograms(
            df_original=self.df,
            variables=variables,
            method=method,
            epsilon=epsilon,
            bins=bins,
            **kwargs
        )