def direct_encoding_privacy(self):
    def direct_encoding(value, d, epsilon):
        p = np.exp(epsilon) / (np.exp(epsilon) + d - 1)
        q = 1 / (np.exp(epsilon) + d - 1)
        
        probabilities = np.full(d, q)
        probabilities[self.category_to_index[value]] = p
        privatized_index = np.random.choice(d, p=probabilities)
        return self.index_to_category[privatized_index]
    
    df_priv = self.df.copy()
    df_priv[self.column] = df_priv[self.column].apply(lambda x: direct_encoding(x, self.d, self.epsilon))
    return df_priv

def optimized_unary_encoding_privacy(self):
    def optimized_unary_encoding(value, d, epsilon):
        p = 0.5
        q = 1 / (np.exp(epsilon) + 1)
        
        binary_vector = np.zeros(d)
        binary_vector[self.category_to_index[value]] = 1
        
        perturbed_vector = np.zeros(d)
        for i in range(d):
            if binary_vector[i] == 1:
                perturbed_vector[i] = np.random.choice([1, 0], p=[p, 1 - p])
            else:
                perturbed_vector[i] = np.random.choice([1, 0], p=[q, 1 - q])
        
        indices_positivos = np.where(perturbed_vector == 1)[0]
        if len(indices_positivos) == 0:
            indices_positivos = [self.category_to_index[value]]
        
        privatized_index = np.random.choice(indices_positivos)
        return self.index_to_category[privatized_index]
    
    df_priv = self.df.copy()
    df_priv[self.column] = df_priv[self.column].apply(lambda x: optimized_unary_encoding(x, self.d, self.epsilon))
    return df_priv

def rappor_privacy(self):
    def rappor_encode(value, d, f=0.5):
        binary_vector = np.zeros(d)
        binary_vector[self.category_to_index[value]] = 1
        
        perturbed_vector = np.zeros(d)
        for i in range(d):
            if binary_vector[i] == 1:
                perturbed_vector[i] = np.random.choice([1, 0], p=[1 - f, f])
            else:
                perturbed_vector[i] = np.random.choice([1, 0], p=[f, 1 - f])
        
        indices_positivos = np.where(perturbed_vector == 1)[0]
        if len(indices_positivos) == 0:
            indices_positivos = [self.category_to_index[value]]
        
        privatized_index = np.random.choice(indices_positivos)
        return self.index_to_category[privatized_index]
    
    df_priv = self.df.copy()
    df_priv[self.column] = df_priv[self.column].apply(lambda x: rappor_encode(x, self.d))
    return df_priv