import torch
import numpy as np
import pandas as pd

class AlgorithmDP:
    @staticmethod
    def duchi_solution(t_i_vector, epsilon):
        """
        Apply the Duchi et al. differential privacy solution to a vector.

        Parameters:
        t_i_vector (list or torch.Tensor): The input vector of numbers.
        epsilon (float): The privacy parameter.

        Returns:
        np.array: The transformed vector.
        """
        # Convert the input vector to a tensor
        t_i_tensor = torch.tensor(t_i_vector, dtype=torch.float32)
        
        # Ensure the input is within the range [-1, 1]
        t_i_tensor = torch.clamp(t_i_tensor, -1, 1)

        # Calculate the probabilities
        e_epsilon = torch.exp(torch.tensor(epsilon, dtype=torch.float32))
        prob = (e_epsilon - 1) / (2 * e_epsilon + 2) * t_i_tensor + 0.5

        # Sample Bernoulli variables
        u = torch.bernoulli(prob.clone().detach())

        # Calculate the output based on the values of u
        t_i_star = torch.where(u == 1, 
                               (e_epsilon + 1) / (e_epsilon - 1),
                               (1 - e_epsilon) / (e_epsilon + 1))
        
        return t_i_star.numpy()

    @staticmethod
    def piecewise_mechanism(t_i_vector, epsilon):
        t_i_tensor = torch.tensor(t_i_vector, dtype=torch.float32)
        t_i_tensor = torch.clamp(t_i_tensor, -1, 1)
        C = (torch.exp(torch.tensor(epsilon / 2)) + 1) / (torch.exp(torch.tensor(epsilon / 2)) - 1)
        def l(t_i): return (C + 1) / 2 * t_i - (C - 1) / 2
        def r(t_i): return l(t_i) + C - 1
        x = torch.rand(t_i_tensor.shape)
        threshold = torch.exp(torch.tensor(epsilon / 2)) / (torch.exp(torch.tensor(epsilon / 2)) + 1)
        t_i_star = torch.empty(t_i_tensor.shape, dtype=torch.float32)
        for i in range(t_i_tensor.shape[0]):
            l_val, r_val = l(t_i_tensor[i]), r(t_i_tensor[i])
            if x[i] < threshold:
                if l_val >= r_val: r_val = l_val + 1e-5
                t_i_star[i] = torch.distributions.Uniform(l_val, r_val).sample()
            else:
                if torch.rand(1) < 0.5:
                    if -C >= l_val: l_val = -C + 1e-5
                    t_i_star[i] = torch.distributions.Uniform(-C, l_val).sample()
                else:
                    if r_val >= C: r_val = C - 1e-5
                    t_i_star[i] = torch.distributions.Uniform(r_val, C).sample()
        return t_i_star.numpy()

    @staticmethod
    def oue_mechanism(value, categories, epsilon):
        value_index = categories.index(value)
        n = len(categories)
        encoded = np.zeros(n)
        encoded[value_index] = 1
        
        p = 1 / (1 + np.exp(epsilon / 2))
        noisy_vector = np.array([bit if np.random.rand() < p else 1 - bit for bit in encoded])
        return noisy_vector

    @staticmethod
    def duchi_multidimensional_solution(t_i_matrix, epsilon):
        """
        Apply the Duchi et al. differential privacy solution to a multidimensional vector.

        Parameters:
        t_i_matrix (list or torch.Tensor): The input matrix of numbers.
        epsilon (float): The privacy parameter.

        Returns:
        np.array: The transformed matrix.
        """
        t_i_tensor = torch.tensor(t_i_matrix, dtype=torch.float32)
        t_i_tensor = torch.clamp(t_i_tensor, -1, 1)
        d = t_i_tensor.shape[1]

        v = torch.empty_like(t_i_tensor).uniform_(-1, 1)
        v = torch.sign(v)

        e_epsilon = torch.exp(torch.tensor(epsilon, dtype=torch.float32))
        threshold = e_epsilon / (e_epsilon + 1)

        u = torch.bernoulli(torch.tensor([threshold]))

        T_plus = (e_epsilon + 1) / (e_epsilon - 1)
        T_minus = (1 - e_epsilon) / (e_epsilon + 1)

        t_i_star = torch.empty_like(t_i_tensor, dtype=torch.float32)

        for i in range(t_i_tensor.shape[0]):
            if u == 1:
                t_i_star[i] = torch.sign(v[i]) * T_plus
            else:
                t_i_star[i] = torch.sign(-v[i]) * T_minus

        return t_i_star.numpy()

    @staticmethod
    def our_method_multidimensional(df, categorical_columns, numerical_columns, epsilon):
        t_i_tensor = torch.tensor(df[numerical_columns].values, dtype=torch.float32)
        t_i_tensor = torch.clamp(t_i_tensor, -1, 1)
        d = len(numerical_columns)
        C = (torch.exp(torch.tensor(epsilon / 2)) + 1) / (torch.exp(torch.tensor(epsilon / 2)) - 1)
        k = max(1, min(d, int(epsilon / 2.5)))
        indices = np.random.choice(d, k, replace=False)
        
        t_i_star = np.zeros((t_i_tensor.shape[0], d))

        for i in range(t_i_tensor.shape[0]):
            for j in indices:
                x_ij = AlgorithmDP.piecewise_mechanism([t_i_tensor[i][j].item()], epsilon / k)[0]
                t_i_star[i][j] = d / k * x_ij

        perturbed_data = pd.DataFrame(t_i_star, columns=numerical_columns)
        
        for col in categorical_columns:
            categories = df[col].unique().tolist()
            perturbed_data[col] = df[col].apply(
                lambda x: AlgorithmDP.oue_mechanism(x, categories, epsilon / k)
            )

        return perturbed_data
