# Differential Privacy Transformer

## Introduction
The Differential Privacy Transformer is a mathematical framework designed to enable the analysis of sensitive data while ensuring individual privacy. This package facilitates the application of differential privacy mechanisms to datasets, supporting both quantitative and categorical variables. It is intended for researchers and professionals who want to incorporate privacy protections into their data analysis workflows.

---

## Overview of the Package
The package provides a primary class, `DifferentialPrivacyTransformer`, that acts as the interface for applying differential privacy mechanisms to a dataset. It includes additional functionality for visualizing and analyzing the impact of privatization, as well as implementations of various differential privacy mechanisms for quantitative and categorical data.

---

## Package Structure

- **`dp_transformer.py`**: Contains the `DifferentialPrivacyTransformer` class.
- **`dp_plotting.py`**: Includes functions for visualizing and analyzing privatized data.
- **`dp_mechanisms/`**: Directory housing differential privacy mechanisms:
  - `quantitative.py`: Mechanisms for quantitative data.
  - `categorical.py`: Mechanisms for categorical data.

---

## DifferentialPrivacyTransformer Class
The `DifferentialPrivacyTransformer` class is the main interface for applying differential privacy mechanisms to a Pandas DataFrame. It supports the transformation of quantitative and categorical variables with adjustable privacy levels via the ε parameter.

### Initialization
```python
DifferentialPrivacyTransformer(
    df,
    epsilon_quantitative=1.0,
    epsilon_categorical=1.0,
    quantitative_vars=None,
    categorical_vars=None
)
```
#### Parameters:
- **`df`** (*pd.DataFrame*): The dataset to be transformed.
- **`epsilon_quantitative`** (*float*): Privacy budget for quantitative variables (default: 1.0).
- **`epsilon_categorical`** (*float*): Privacy budget for categorical variables (default: 1.0).
- **`quantitative_vars`** (*list of str*): Names of quantitative variables. Automatically detected if `None`.
- **`categorical_vars`** (*list of str*): Names of categorical variables. Automatically detected if `None`.

### Main Methods
#### `fit_quantitative`
Applies differential privacy mechanisms to the specified quantitative variables.
```python
fit_quantitative(
    quantitative_vars=None,
    method='duchi',
    epsilon=None
)
```
#### `fit_categorical`
Applies differential privacy mechanisms to the specified categorical variables.
```python
fit_categorical(
    categorical_vars=None,
    method='direct_encoding',
    epsilon=None
)
```
#### `calculate_utility_metrics`
Calculates utility metrics to evaluate the impact of privatization on quantitative variables.
```python
calculate_utility_metrics(variables=None)
```
#### Visualization Methods
- **`plot_mean_vs_privatized`**: Generates a plot comparing the mean of original variables to privatized ones for different ε values.
- **`plot_histograms`**: Generates comparative histograms for original and privatized variables.

---

## Differential Privacy Mechanisms

### For Quantitative Data
Implemented in `dp_mechanisms/quantitative.py`:
- **Duchi Mechanism** (`'duchi'`): Suitable for data in the range [-1, 1].
- **Laplace Mechanism** (`'laplace'`): Adds Laplacian noise proportional to the data’s sensitivity.
- **Piecewise Mechanism** (`'piecewise'`): Divides the domain into segments and applies noise accordingly.
- **Multidimensional Duchi Mechanism** (`'multidimensional_duchi'`): Extension for multidimensional data.
- **Custom Multidimensional Mechanism** (`'multidimensional'`): Selectively applies unidimensional mechanisms to random dimensions.

### For Categorical Data
Implemented in `dp_mechanisms/categorical.py`:
- **Direct Encoding** (`'direct_encoding'`): Assigns probabilities to each category and applies randomized response.
- **Optimized Unary Encoding (OUE)** (`'oue'`): Uses unary encoding with optimized perturbation.
- **RAPPOR** (`'rappor'`): Employs unary encoding with multiple rounds of perturbation.

---

## Usage Examples

### Quantitative Variables
```python
import pandas as pd
import seaborn as sns
from dp_transformer import DifferentialPrivacyTransformer

# Load Titanic dataset
df = sns.load_dataset('titanic').dropna()
quantitative_vars = ['age', 'fare']

# Instantiate the transformer
dp_transformer = DifferentialPrivacyTransformer(
    df,
    epsilon_quantitative=1.0,
    quantitative_vars=quantitative_vars
)

# Apply Duchi mechanism
df_priv = dp_transformer.fit_quantitative(method='duchi')

# Calculate utility metrics
metrics = dp_transformer.calculate_utility_metrics()
print(metrics)

# Plot original vs privatized means
dp_transformer.plot_mean_vs_privatized(
    variables=quantitative_vars,
    method='duchi',
    epsilons=[0.1, 0.5, 1.0]
)
```

### Categorical Variables
```python
# Define categorical variables
categorical_vars = ['sex', 'class', 'embarked']

# Instantiate the transformer
dp_transformer = DifferentialPrivacyTransformer(
    df,
    epsilon_categorical=1.0,
    categorical_vars=categorical_vars
)

# Apply direct encoding
df_priv = dp_transformer.fit_categorical(method='direct_encoding')

# Display transformed data
print(df_priv[categorical_vars].head())
```

---

## Contact
**Authors:**
- Luisa María De La Hortúa (lm.delahortua@uniandes.edu.co)
- David Moreno (ds.morenom1@uniandes.edu.co)
- Allan Ramírez (as.ramirez2@uniandes.edu.co)
- David Romero (ds.romerog1@uniandes.edu.co)

---

This README provides an overview of the Differential Privacy Transformer and its functionality. For detailed examples and more information, refer to the documentation.