# Copyright (c) 2025 Hehui Zheng, ETH Zurich.
# Licensed under the MIT License. See LICENSE file for details.

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze, unfreeze
import numpy as np

class MinMaxNormalizer:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.feature_range = (-1, 1)

    def normalize(self, x):
        x_std = (x - self.min_val) / (self.max_val - self.min_val)
        return x_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def denormalize(self, x):
        x_std = (x - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        return x_std * (self.max_val - self.min_val) + self.min_val
    

class MuscleInputNormalizer:
    def __init__(self):
        self.normalizer_volt = MinMaxNormalizer(1, 7)
        self.normalizer_disp = MinMaxNormalizer(-12.71896863, 0.01398294)
        self.normalizer_vel = MinMaxNormalizer(-25.94804818, 32.10096358)

    def normalize(self, volt, disp, vel, convert_unit=False):
        if convert_unit:
            volt, disp, vel = self.convert_unit(volt, disp, vel)
        return (self.normalizer_volt.normalize(volt), self.normalizer_disp.normalize(disp), self.normalizer_vel.normalize(vel))
    
    def denormalize(self, normalize_volt, normalize_disp, normalize_vel, convert_unit=False):
        volt = self.normalizer_volt.denormalize(normalize_volt)
        disp = self.normalizer_disp.denormalize(normalize_disp)
        vel = self.normalizer_vel.denormalize(normalize_vel)
        if convert_unit:
            volt, disp, vel = self.convert_unit_to_SI(volt, disp, vel)
        return (volt, disp, vel)

    
    def convert_unit(self, volt, disp, vel):
        # Convert SI units to real world units
        # displacement: m -> mm
        # velocity: m/s -> mm/s
        return (volt, disp*1000, vel*1000)

    def convert_unit_to_SI(self, volt, disp, vel):
        # Convert real world units to SI units
        # displacement: mm -> m
        # velocity: mm/s -> m/s
        return (volt, disp/1000, vel/1000)
    

class MuscleOutputNormalizer:
    def __init__(self):
        self.F_min = -7.26166209
        self.F_max = 0.0
        self.normalizer_F = MinMaxNormalizer(self.F_min, self.F_max)

    def normalize(self, F):
        # Normalize the force
        return self.normalizer_F.normalize(F)

    def denormalize(self, F):
        F_pred = self.normalizer_F.denormalize(F)
        # Clip the force to the min and max values
        return np.clip(F_pred, -50, 0)


class PySRModel():
    def __init__(self):
        pass
    def apply(self, variables, X, train=False):
        # X = np.array([volt, disp, vel])
        return jnp.array([jnp.sin((X[0] + X[1]*(X[0] + 1.3307986) + X[2])*0.4780912 + 2.039332)])

class PySRModel_200Hz():
    def __init__(self):
        pass
    def apply(self, variables, X, train=False):
        # X = np.array([volt, disp, vel])
        return jnp.array([11.854346**(X[0] + X[1] + (X[2] - jnp.abs(X[2]) - jnp.abs(X[0] + jnp.abs(X[1]) - 0.4742842))*0.44613895)/(-10.574492) + 0.99724096])
    

class MLPModel(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int
    dropout_prob: float = 0.0
    deterministic: bool = True  # Controls dropout during training/inference

    @nn.compact
    def __call__(self, X, train=True):
        assert self.num_layers >= 2, "Number of layers must be at least 2"

        X = nn.Dense(self.hidden_dim)(X)
        X = nn.relu(X)
        # X = nn.Dropout(rate=self.dropout_prob, deterministic=not train)(X)

        for _ in range(self.num_layers - 2):
            X = nn.Dense(self.hidden_dim)(X)
            X = nn.relu(X)
            # X = nn.Dropout(rate=self.dropout_prob, deterministic=not train)(X)

        X = nn.Dense(self.output_dim)(X)
        return X

    @classmethod
    def create_model(cls, rng, input_dim, hidden_dim, output_dim, num_layers, dropout_prob=0.0):
        """Class method to initialize the model and parameters."""
        model = cls(input_dim, hidden_dim, output_dim, num_layers, dropout_prob)
        variables = model.init(rng, jnp.ones((1, input_dim)), train=False)
        return model, variables

    @staticmethod
    def load_pytorch_param(flax_variables, path):
        numpy_params = np.load(path)
        # Convert to JAX format
        jax_params = {k: jnp.array(v) for k, v in numpy_params.items()}
        
        # Convert FrozenDict to mutable dictionary
        flax_params = unfreeze(flax_variables['params'])

        # Manually map PyTorch weights to Flax model
        for name, param in jax_params.items():
            # Assign parameters to Flax model
            keys = name.split('.')
            jax_keys = ['Dense_'+str(int(keys[1])//3), keys[2].replace("weight", "kernel").replace("bias", "bias")]

            if "weight" in name:
                param = param.T  # PyTorch uses (out, in), Flax uses (in, out)

            # Assign parameters to Flax model
            sub_dict = flax_params
            for k in jax_keys[:-1]:
                sub_dict = sub_dict[k]
            sub_dict[jax_keys[-1]] = jnp.array(param)

        # Convert back to FrozenDict
        return freeze({'params': flax_params}) 


class LSTMModel(nn.Module):
    input_dim: int
    hidden_dim: int
    num_layers: int
    output_dim: int
    dropout_prob: float

    def setup(self):
        self.lstm_cells = [
            nn.OptimizedLSTMCell(features=self.hidden_dim) for _ in range(self.num_layers)
        ]
        self.fc = nn.Dense(features=self.output_dim)

    def __call__(self, x, carry=None, train=False):
        if carry is None:
            carry = self.initialize_carry(jax.random.PRNGKey(0), x.shape[0], self.hidden_dim, self.num_layers)
        outputs = []
        for t in range(x.shape[1]):  # Iterate over the sequence length
            for i, cell in enumerate(self.lstm_cells):
                carry[i], y = cell(carry[i], x[:, t, :] if i == 0 else y)
            outputs.append(y)
        y = jnp.stack(outputs, axis=1)
        y = self.fc(y[:, -1, :])  # Apply fully connected layer on the last hidden state
        return y, carry

    @staticmethod
    def initialize_carry(rng, batch_size, hidden_dim, num_layers):
        return [
            nn.OptimizedLSTMCell(features=hidden_dim).initialize_carry(
                rng, (batch_size, hidden_dim)
            )
            for _ in range(num_layers)
        ]

    @classmethod
    def create_model(cls, rng, input_dim, hidden_dim, output_dim, num_layers, dropout_prob):
        model = cls(input_dim, hidden_dim, num_layers, output_dim, dropout_prob)
        batch_size = 1
        x = jnp.ones((batch_size, 10, input_dim))  # Example input
        carry = cls.initialize_carry(rng, batch_size, hidden_dim, num_layers)
        variables = model.init(rng, x, carry)
        return model, variables


    def load_pytorch_param(self, flax_variables, path):
        """Loads PyTorch-trained weights into the JAX model."""
        # Load the saved NumPy parameters
        numpy_params = np.load(path)
        # Convert to JAX format
        # jax_params = {k: jnp.array(v) for k, v in numpy_params.items()}
        # Convert FrozenDict to mutable dictionary
        flax_params = unfreeze(flax_variables['params'])

        num_layers = len([k for k in numpy_params.keys() if "rnn.weight_ih_l" in k])

        for layer in range(num_layers):
            # Extract PyTorch LSTM weights and biases
            weight_ih = numpy_params[f"rnn.weight_ih_l{layer}"]  # Shape (4 * hidden_dim, input_dim)
            weight_hh = numpy_params[f"rnn.weight_hh_l{layer}"]  # Shape (4 * hidden_dim, hidden_dim)
            bias_ih = numpy_params[f"rnn.bias_ih_l{layer}"]  # Shape (4 * hidden_dim,)
            bias_hh = numpy_params[f"rnn.bias_hh_l{layer}"]  # Shape (4 * hidden_dim,)

            # Split into four gates (input, forget, cell, output)
            weight_ih_split = np.split(weight_ih, 4, axis=0)  # i, f, g, o
            weight_hh_split = np.split(weight_hh, 4, axis=0)
            bias_ih_split = np.split(bias_ih, 4, axis=0)
            bias_hh_split = np.split(bias_hh, 4, axis=0)

            # Assign to Flax parameters with proper transpositions
            flax_params[f"lstm_cells_{layer}"] = {
                "ii": {"kernel": jnp.array(weight_ih_split[0].T)},
                "if": {"kernel": jnp.array(weight_ih_split[1].T)},
                "ig": {"kernel": jnp.array(weight_ih_split[2].T)},
                "io": {"kernel": jnp.array(weight_ih_split[3].T)},
                "hi": {"kernel": jnp.array(weight_hh_split[0].T), "bias": jnp.array(bias_hh_split[0]) + jnp.array(bias_ih_split[0])},
                "hf": {"kernel": jnp.array(weight_hh_split[1].T), "bias": jnp.array(bias_hh_split[1]) + jnp.array(bias_ih_split[1])},
                "hg": {"kernel": jnp.array(weight_hh_split[2].T), "bias": jnp.array(bias_hh_split[2]) + jnp.array(bias_ih_split[2])},
                "ho": {"kernel": jnp.array(weight_hh_split[3].T), "bias": jnp.array(bias_hh_split[3]) + jnp.array(bias_ih_split[3])},
            }

        # Fully connected (FC) layer (PyTorch stores as (out_dim, hidden_dim), Flax expects (hidden_dim, out_dim))
        flax_params["fc"] = {
            "kernel": jnp.array(numpy_params["fc.weight"].T),  # Transpose for Flax format
            "bias": jnp.array(numpy_params["fc.bias"]),
        }

        return freeze({'params': flax_params})


class ResidualModel(nn.Module):
    # Define module hyperparameters as fields using type annotations.
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int
    dropout_prob: float = 0.0

    def setup(self):
        # Define constant attributes and child modules here.
        self.epsilon0 = 8.854e-12  # F/m
        self.epsilonr = jnp.array(0.29535)
        self.alpha0 = jnp.array(0.19213)
        self.w = jnp.array(0.09)
        self.t = jnp.array(1.2e-05)
        self.Lp = jnp.array(0.02)

        self.mlp = MLPModel(self.input_dim, self.hidden_dim, self.output_dim,
                            self.num_layers, self.dropout_prob)
        self.input_normalizer = MuscleInputNormalizer()
        self.output_normalizer = MuscleOutputNormalizer()

    def x2alpha(self, denormalize_disp):
        return jnp.sinh(jnp.log(1.41220140107638*(1 - 0.371358817510341*denormalize_disp)**0.3484297)**1.5195487)

    @nn.compact
    def __call__(self, X, train=False):
        assert self.num_layers >= 2, "Number of layers must be at least 2"

        if isinstance(X, np.ndarray):
            X = jnp.array(X)
        X = X.reshape(1, -1)

        # Predict the force using the analytical model
        kvolt, disp, _ = self.input_normalizer.denormalize(*X[0], convert_unit=False)  # keep unit in mm, do not convert back to SI
        # print('disp:', disp)
        alpha = self.x2alpha(disp)
        # print('alpha:', alpha)
        alpha_clamp = jnp.clip(alpha, 0.01, np.pi/2)
        volt = 1e3 * kvolt  # kilo Volts -> Volts
        # print('Voltage:', volt)
        force_analytical = -self.F(volt, alpha_clamp)
        # print('force_analytical:', force_analytical)
        yPred = self.output_normalizer.normalize(force_analytical)
        # print('yPred:', yPred)

        # Predict the residual using the MLP
        yRes = self.mlp(X)
        # print('yRes:', yRes)
        return yPred + yRes

    def F(self, V, alpha):
        return (self.w / (4 * self.t)) * jnp.cos(alpha) / (1 - jnp.cos(alpha)) \
               * self.epsilon0 * self.epsilonr * V**2

    @classmethod
    def create_model(cls, rng, input_dim, hidden_dim, output_dim, num_layers, dropout_prob=0.0):
        """Class method to initialize the model and parameters."""
        model = cls(input_dim, hidden_dim, output_dim, num_layers, dropout_prob)
        variables = model.init(rng, jnp.ones((1, input_dim)), train=False)
        return model, variables

    def load_pytorch_param(self, flax_variables, path):
        # Convert FrozenDict to mutable dictionary
        mlp_flax_params = unfreeze(flax_variables['params']['mlp'])
        loaded_mlp_flax_variables = MLPModel.load_pytorch_param(freeze({'params': mlp_flax_params}), path)
        loaded_mlp_flax_params = unfreeze(loaded_mlp_flax_variables['params'])
        return freeze({'params': {'mlp': loaded_mlp_flax_params}})

def forward(variables, model, X, carry=None, train=False):
    if carry is None:
        return model.apply(variables, X, train=train)
    return model.apply(variables, X, carry, train=train)
