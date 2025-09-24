import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import optax
import pgx
from common.mlp import MLP
from flax.training.train_state import TrainState
from omegaconf import DictConfig


class Agent(nn.Module):
    state_dim: int
    action_dim: int
    hidden_dims: list[int]
    activation: nn.Module

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[distrax.Distribution, jnp.ndarray]:
        logits = MLP(
            dims=[self.state_dim, *self.hidden_dims, self.action_dim],
            activation=self.activation,
        )(x)
        action = distrax.Categorical(logits=logits)

        value = MLP(
            dims=[self.state_dim, *self.hidden_dims, 1],
            activation=self.activation,
        )(x)

        return action, value.squeeze(-1)


@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig):
    rng = jax.random.PRNGKey(cfg.seed)

    env = pgx.make("2048")

    def linear_schedule(count: int) -> float:
        return 1e-3

    agent = Agent(
        state_dim=env.observation_shape,
        action_dim=env.num_actions,
        hidden_dims=[64, 64],
        activation=nn.relu,
    )
    rng, _rng = jax.random.split(rng)
    init_state = jnp.zeros(env.observation_shape)
    network_params = agent.init(rng, init_state)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=linear_schedule, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=agent.apply,
        params=network_params,
        tx=tx,
    )


if __name__ == "__main__":
    train()
