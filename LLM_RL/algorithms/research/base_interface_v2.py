from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from transformers.generation import GenerationConfig
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as PS
from jaxtyping import PyTree
from flax import struct
from functools import partial
from typing import List, Optional, Union, Tuple, Callable, NamedTuple
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase
from JaxSeq.utils import with_named_sharding_constraint, match_partition_rules, BlockingStrategy, block_sequences, Padding, Truncation
from optax import softmax_cross_entropy_with_integer_labels, softmax_cross_entropy
from flax.training.train_state import TrainState
from transformers.modeling_flax_outputs import FlaxCausalLMOutput
from transformers.generation import FlaxBeamSearchOutput, FlaxGreedySearchOutput, FlaxSampleOutput
from flax.core import FrozenDict, freeze
from jax.experimental.pjit import pjit
from LLM_RL.utils import get_tensor_stats


def loss_fn_mask(
    model: FlaxPreTrainedModel, 
    params: PyTree,
    target_params: Optional[PyTree],
    pi_beta_params: PyTree,
    input_ids: jax.Array, 
    input_attention_mask: jax.Array, 
    input_position_ids: jax.Array, 
    input_training_mask: jax.Array,
    rewards: jax.Array,
    gammas: jax.Array,
    prng_key: jax.random.PRNGKeyArray, 
    train: bool,
    pad_token_id: Optional[int]=None,
) -> Tuple[jax.Array, PyTree]:
    
    # current policy 
    model_output = model(
        input_ids=input_ids, 
        attention_mask=input_attention_mask, 
        position_ids=input_position_ids, 
        params=params, 
        dropout_rng=prng_key, 
        train=train, 
    )

    if target_params is not None:
        prng_key, new_key = jax.random.split(prng_key)
        target_model_output = model(
            input_ids=input_ids, 
            attention_mask=input_attention_mask, 
            position_ids=input_position_ids, 
            params=target_params, 
            dropout_rng=new_key, 
            train=False, 
        )
    else:
        target_model_output = model_output

    # behavior policy outputs 
    prng_key, new_key = jax.random.split(prng_key)
    pi_beta_model_output = model(
        input_ids=input_ids, 
        attention_mask=input_attention_mask, 
        position_ids=input_position_ids, 
        params=pi_beta_params, 
        dropout_rng=new_key, 
        train=train, 
    )

    logits = model_output.logits[:, :-1, :].astype(jnp.float32)
    pi_beta_logits = pi_beta_model_output.logits[:, :-1, :].astype(jnp.float32)
    target_logits = target_model_output.logits[:, 1:, :].astype(jnp.float32)
    # target_pi_beta_logits = pi_beta_model_output.logits[:, 1:, :].astype(jnp.float32)

    # Compute bellman target values
    target_ids = input_ids[:, 1:]
    targets = jax.lax.stop_gradient(
        rewards[:, :-1] + gammas[:, :-1] *
        jnp.max(
            (jax.nn.softmax(pi_beta_logits) > 1e-4).astype(jnp.float32) *  # Clipping based on pi_beta probabilities
            jnp.exp(jax.nn.log_softmax(target_logits) - jax.nn.log_softmax(pi_beta_logits)), axis=-1)
    )

    # Compute smooth distribution
    vocab_size = logits.shape[-1]
    target_ids_one_hot = jax.nn.one_hot(target_ids, num_classes=vocab_size, dtype=jnp.float32, axis=-1)
    assert target_ids_one_hot.shape == logits.shape
    # if pad_token_id is provided, put rest of weight on pad_token, else distribute evenly among rest
    # of tokens
    if pad_token_id is not None:
        pad_one_hot = jnp.zeros_like(logits).at[:, :, pad_token_id].set(1)
        target_distribution = (
            targets[..., None] * target_ids_one_hot + (1 - targets[..., None]) * pad_one_hot
        )
    else:
        target_distribution = (
            targets[..., None] * target_ids_one_hot +
            (1 - targets[..., None]) / (vocab_size - 1) * (jnp.ones_like(logits) - target_ids_one_hot)
        )

    mask = input_attention_mask[:, 1:]
    q_loss = softmax_cross_entropy(logits, target_distribution) * mask
    bc_loss = softmax_cross_entropy_with_integer_labels(pi_beta_logits, target_ids) * mask
    # q_bc_loss = softmax_cross_entropy(logits, target_ids) * mask
    loss = (q_loss + bc_loss).sum() / mask.sum()

    mask = input_attention_mask[:, 1:]
    n = mask.sum()
    q_loss = softmax_cross_entropy(logits, target_distribution) * mask
    q_loss = q_loss.sum() / n
    bc_loss = softmax_cross_entropy_with_integer_labels(pi_beta_logits, target_ids) * mask
    bc_loss = bc_loss.sum() / n
    loss = q_loss + bc_loss

    agent_mask = input_training_mask[:, 1:]
    logs = dict(
        losses=dict(
            total_loss=loss,
            q_loss=q_loss,
            bc_loss=bc_loss
        ),
        agent_targets=get_tensor_stats(targets, mask=mask * agent_mask, n=n),
        env_targets=get_tensor_stats(targets, mask=mask * jnp.invert(agent_mask), n=n),
        rewards=get_tensor_stats(rewards, mask=mask, n=n),
    )
    return loss, logs
