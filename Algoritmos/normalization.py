import torch

class Normalizer:
    @staticmethod
    def normalize_to_range(vector, min_range=-1, max_range=1):
        """
        Normalize a vector of numbers to a specified range using PyTorch.

        Parameters:
        vector (list or torch.Tensor): The vector of numbers to normalize.
        min_range (float): The minimum value of the range. Default is -1.
        max_range (float): The maximum value of the range. Default is 1.

        Returns:
        torch.Tensor: The normalized vector.
        """
        vector = torch.tensor(vector, dtype=torch.float32)
        min_val = torch.min(vector)
        max_val = torch.max(vector)

        # Normalize to [0, 1]
        normalized = (vector - min_val) / (max_val - min_val)

        # Scale to [min_range, max_range]
        normalized = normalized * (max_range - min_range) + min_range

        return normalized
