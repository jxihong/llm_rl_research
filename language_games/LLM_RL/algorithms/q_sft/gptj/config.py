# coding=utf-8
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GPT-J model configuration"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from jax.sharding import PartitionSpec as PS
import re
import jax
from typing import Optional, Dict, Any

logger = logging.get_logger(__name__)


class GPTJConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GPTJModel`]. It is used to instantiate a GPT-J
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the GPT-J
    [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B) architecture. Configuration objects inherit from
    [`PretrainedConfig`] and can be used to control the model outputs. Read the documentation from [`PretrainedConfig`]
    for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 50400):
            Vocabulary size of the GPT-J model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPTJModel`].
        n_positions (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        rotary_dim (`int`, *optional*, defaults to 64):
            Number of dimensions in the embedding that Rotary Position Embedding is applied to.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu_new"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    Example:
    ```python
    >>> from transformers import GPTJModel, GPTJConfig
    >>> # Initializing a GPT-J 6B configuration
    >>> configuration = GPTJConfig()
    >>> # Initializing a model from the configuration
    >>> model = GPTJModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "gptj"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50400,
        n_positions=2048,
        n_embd=4096,
        n_layer=28,
        n_head=16,
        rotary_dim=64,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=False,
        gradient_checkpointing=True, 
        gradient_checkpointing_policy='nothing_saveable', 
        unpadded_vocab_size=None, 
        mesh: Optional[jax.sharding.Mesh]=None, 
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.rotary_dim = rotary_dim
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_checkpointing_policy = gradient_checkpointing_policy
        self.unpadded_vocab_size = unpadded_vocab_size
        if self.unpadded_vocab_size is None:
            self.unpadded_vocab_size = self.vocab_size

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mesh = mesh

        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )
    
    @staticmethod
    def get_partition_rules():
        return [
            # embeddings
            (re.escape("['transformer']['wte']['embedding']"), PS("mp", "fsdp")), 
            # atention
            (''.join((re.escape("['attn']"), r"\['(k_proj|q_proj|v_proj)'\]", re.escape("['kernel']"))), PS("fsdp", "mp")), 
            (re.escape("['attn']['out_proj']['kernel']"), PS("mp", "fsdp")), 
            # mlp
            (re.escape("['mlp']['fc_in']['kernel']"), PS("fsdp", "mp")), 
            (re.escape("['mlp']['fc_in']['bias']"), PS("mp")), 
            (re.escape("['mlp']['fc_out']['kernel']"), PS("mp", "fsdp")), 
            (re.escape("['mlp']['fc_out']['bias']"), PS()), 
            # layer norms
            ((r"\['ln_\d+'\]\['bias'\]"), PS()), 
            ((r"\['ln_\d+'\]\['scale'\]"), PS()), 
            (re.escape("['ln_f']['bias']"), PS()), 
            (re.escape("['ln_f']['scale']"), PS()), 
            # output head
            (re.escape("['lm_head']['kernel']"), PS("fsdp", "mp")), 
            (re.escape("['lm_head']['bias']"), PS("mp")), 
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        if self.mesh is None:
            return super().to_dict()
        else:
            new_conf = GPTJConfig(**self.__dict__)
            new_conf.mesh = None
            return new_conf.to_dict()
