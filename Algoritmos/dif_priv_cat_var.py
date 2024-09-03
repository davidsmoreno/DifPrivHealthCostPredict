import numpy as np
import pandas as pd
import torch
import time

# Decorador para medir el tiempo de ejecución
def medir_tiempo(func):
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        fin = time.time()
        tiempo_total = fin - inicio
        print(f"El método '{func.__name__}' tomó {tiempo_total:.4f} segundos en ejecutarse.")
        return resultado
    return wrapper

# Clase Padre con Numpy
class PrivacidadDiferencialCategoricaNumpy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    @medir_tiempo
    def aplicar_ruido(self, conteos):
        ruido = np.random.laplace(0, 1/self.epsilon, size=conteos.shape)
        conteos_con_ruido = conteos + ruido
        return np.maximum(conteos_con_ruido, 0)

    @medir_tiempo
    def anonimizar_categoria(self, df, columna):
        conteos = df[columna].value_counts().values
        categorias = df[columna].value_counts().index
        conteos_con_ruido = self.aplicar_ruido(conteos)
        
        df_anonimizado = df.copy()
        df_anonimizado[columna] = df_anonimizado[columna].apply(
            lambda x: np.random.choice(categorias, p=conteos_con_ruido/np.sum(conteos_con_ruido))
        )
        
        return df_anonimizado


# Clase Hija con PyTorch
class PrivacidadDiferencialCategoricaTensor(PrivacidadDiferencialCategoricaNumpy):
    
    @medir_tiempo
    def aplicar_ruido(self, conteos):
        ruido = torch.distributions.Laplace(0, 1/self.epsilon).sample(conteos.shape)
        conteos_con_ruido = conteos + ruido
        return torch.clamp(conteos_con_ruido, min=0)

    @medir_tiempo
    def anonimizar_categoria(self, tensor, categorias):
        conteos = torch.tensor([(tensor == cat).sum().item() for cat in categorias])
        conteos_con_ruido = self.aplicar_ruido(conteos)
        
        probabilidades = conteos_con_ruido / torch.sum(conteos_con_ruido)
        categorias_anonimizadas = torch.multinomial(probabilidades, tensor.shape[0], replacement=True)
        tensor_anonimizado = categorias[categorias_anonimizadas]
        
        return tensor_anonimizado


# Ejemplo de uso con Numpy (DataFrames)
df = pd.DataFrame({
    'categoria': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C', 'A'],
    'valor': [10, 20, 30, 40, 50, 60, 70, 80, 90]
})

privacidad_numpy = PrivacidadDiferencialCategoricaNumpy(epsilon=0.5)
df_anonimizado_numpy = privacidad_numpy.anonimizar_categoria(df, 'categoria')
print("DataFrame anonimizado con Numpy:")
print(df_anonimizado_numpy)

# Ejemplo de uso con PyTorch (Tensores)
tensor_categorico = torch.tensor([0, 1, 0, 2, 1, 0, 2, 2, 0])
categorias_unicas = torch.tensor([0, 1, 2])

privacidad_tensor = PrivacidadDiferencialCategoricaTensor(epsilon=0.5)
tensor_anonimizado_tensor = privacidad_tensor.anonimizar_categoria(tensor_categorico, categorias_unicas)
print("Tensor anonimizado con PyTorch:")
print(tensor_anonimizado_tensor)


