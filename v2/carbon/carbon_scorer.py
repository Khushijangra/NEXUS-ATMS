import torch

class CarbonScorer:
    """
    Computes a carbon emission estimate scalar (Ct) for an intersection.
    Equation: CO2 = α * queue + β * delay + γ * speed_variance
    """
    def __init__(self, alpha=0.5, beta=0.2, gamma=0.3):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def score(self, queue: float, delay: float, speed_variance: float = 0.0) -> torch.Tensor:
        """
        Computes the carbon state based on current traffic parameters.
        Returns tensor of shape (1, 1)
        """
        co2 = (self.alpha * queue) + (self.beta * delay) + (self.gamma * speed_variance)
        
        # Format as SPGRL Ct tensor: (Batch=1, 1)
        Ct = torch.tensor([[co2]], dtype=torch.float32)
        return Ct
