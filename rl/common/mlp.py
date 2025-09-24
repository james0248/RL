import jax.numpy as jnp
import flax.linen as nn


class MLP(nn.Module):
    dims: list[int]
    activation: nn.Module
    output_activation: nn.Module | None = None
    normalization: nn.Module | None = None
    pre_normalization: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for dim in self.dims[:-1]:
            if self.normalization is not None and self.pre_normalization:
                x = self.normalization(x)

            x = self.activation(nn.Dense(dim)(x))

            if self.normalization is not None and not self.pre_normalization:
                x = self.normalization(x)

        x = nn.Dense(self.dims[-1])(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x
