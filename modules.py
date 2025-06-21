import numpy as np
from typing import Callable, Optional, Union


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def init_xavier_weights(d_input, d_output):
    return np.random.randn(d_output, d_input) * np.sqrt(
            2.0 / (d_input + d_output)
        ) 


class Sigmoid:
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self._sigmoid(grad) * (1 - self._sigmoid(grad)) * grad

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return np.where(grad > 0, 1, 0) * grad

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class MSE:
    def __init__(self) -> None:
        self.pred: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self.pred, self.target = pred, target
        return np.mean((pred - target) ** 2)

    def backward(self) -> np.ndarray:
        return np.mean(0.5 * (self.pred - self.target)).reshape(1)

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float:
        return self.forward(pred, target)


class SGD:
    def __init__(
        self, params: list, criterion: Callable, lr: float = 0.01, momentum: float = 0.9
    ):
        self.params = params
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        self.updates = [
            [np.zeros_like(layer.W), np.zeros_like(layer.b)] for layer in self.params
        ]

    def zero_grad(self) -> None:
        for layer in self.params:
            layer.dW = 0.0
            layer.db = 0.0

    def step(self) -> None:
        for layer, update in zip(self.params, self.updates):
            update[0] = self.lr * layer.dW + self.momentum * update[0]
            update[1] = self.lr * layer.db + self.momentum * update[1]
            layer.W -= update[0]
            layer.b -= update[1]

    def __call__(self) -> None:
        return self.step()


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self):
        """Return a list of layers that have a Weight vectors"""
        params = []
        for layer in self.layers:
            if hasattr(layer, "W"):
                params.append(layer)
        return params

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x


class LinearLayer:
    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.W: np.ndarray = np.random.randn(out_dim, in_dim) * np.sqrt(
            2.0 / (in_dim + out_dim)
        )  # xavier init
        self.b: np.ndarray = np.zeros(out_dim)
        self.dW: Union[float, np.ndarray] = 0.0
        self.db: Union[float, np.ndarray] = 0.0
        self.x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # print(f"x: {x}, self.W {self.W}, self.b {self.b}")
        self.x = x
        return self.W @ x + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # print(f"shape of incoming grad {grad} \n shape of W {self.W.shape}")
        self.dW = np.outer(grad, self.x)
        self.db = grad
        grad = self.W.T @ grad
        return grad

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class InputEmbeddings:
    pass


class PositionalEncoding:
    pass


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisble by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # using d_k also for d_v, and d_q as they are equal
        self.W_q = LinearLayer(d_model, self.d_k) 
        self.W_k = LinearLayer(d_model, self.d_k)
        self.W_v = LinearLayer(d_model, self.d_k)
        self.W_o = LinearLayer(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V):
        # TODO masking?
        attention_scores = (Q @ K.T) / np.sqrt(self.d_k)
        attention_probs = softmax(attention_scores)
        output = attention_probs @ V 
        return output
    
    def split_heads(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        return Q, K, V     
        
    def combine_heads(self, x):
        pass
    
    def forward(self, x):
        Q, K, V = self.split_heads(x)
        attention_output = self.scaled_dot_product_attention(Q, K, V)
        output = self.combine_heads(attention_output)
        return output
    
    
class FeedForwardLayer:
    def __init__(self, d_model, d_ff):
        self.layer1 = LinearLayer(d_model, d_ff)
        self.activation = ReLU()
        self.layer2 = LinearLayer(d_ff, d_model)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        return self.layer(x)
    


class LayerNormalization:
    pass


class ResidualConnection:
    pass

