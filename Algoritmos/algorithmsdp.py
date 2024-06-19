import torch

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
        """
        Apply the Piecewise Mechanism for One-Dimensional Numeric Data.

        Parameters:
        t_i_vector (list or torch.Tensor): The input vector of numbers in the range [-1, 1].
        epsilon (float): The privacy parameter.

        Returns:
        np.array: The transformed vector within the range [-C, C].
        """
        # Convert the input vector to a tensor
        t_i_tensor = torch.tensor(t_i_vector, dtype=torch.float32)
        
        # Ensure the input is within the range [-1, 1]
        t_i_tensor = torch.clamp(t_i_tensor, -1, 1)
        
        # Calculate C
        C = (torch.exp(torch.tensor(epsilon / 2)) + 1) / (torch.exp(torch.tensor(epsilon / 2)) - 1)
        
        # Define l(t_i) and r(t_i)
        def l(t_i):
            return (C + 1) / 2 * t_i - (C - 1) / 2

        def r(t_i):
            return l(t_i) + C - 1

        # Sample x uniformly at random from [0, 1]
        x = torch.rand(t_i_tensor.shape)

        # Calculate the threshold
        threshold = torch.exp(torch.tensor(epsilon / 2)) / (torch.exp(torch.tensor(epsilon / 2)) + 1)
        
        # Initialize t_i_star
        t_i_star = torch.empty(t_i_tensor.shape, dtype=torch.float32)
        
        # Compute t_i_star based on the condition
        for i in range(t_i_tensor.shape[0]):
            l_val = l(t_i_tensor[i])
            r_val = r(t_i_tensor[i])
            #print(f"t_i_tensor[{i}] = {t_i_tensor[i]}, l(t_i) = {l_val}, r(t_i) = {r_val}, C = {C}")
            
            if x[i] < threshold:
                # Ensure low < high for the uniform distribution
                if l_val >= r_val:
                    r_val = l_val + 1e-5
                # Sample t_i_star uniformly at random from [l(t_i), r(t_i)]
                t_i_star[i] = torch.distributions.Uniform(l_val, r_val).sample()
            else:
                # Sample t_i_star uniformly at random from [-C, l(t_i)) ∪ (r(t_i), C]
                if torch.rand(1) < 0.5:
                    # Ensure low < high for the uniform distribution
                    if -C >= l_val:
                        l_val = -C + 1e-5
                    t_i_star[i] = torch.distributions.Uniform(-C, l_val).sample()
                else:
                    # Ensure low < high for the uniform distribution
                    if r_val >= C:
                        r_val = C - 1e-5
                    t_i_star[i] = torch.distributions.Uniform(r_val, C).sample()

        return t_i_star.numpy()
