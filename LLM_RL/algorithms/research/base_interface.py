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
from LLM_RL.utils import get_tensor_stats
from flax.training.train_state import TrainState
from transformers.modeling_flax_outputs import FlaxCausalLMOutput
from transformers.generation import FlaxBeamSearchOutput, FlaxGreedySearchOutput, FlaxSampleOutput
from flax.core import FrozenDict, freeze
from jax.experimental.pjit import pjit


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
    pad_token_id: Optional[int]=None
) -> Tuple[jax.Array, PyTree]:
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
    target_pi_beta_logits = pi_beta_model_output.logits[:, 1:, :].astype(jnp.float32)

    # Compute bellman target values
    target_ids = input_ids[:, 1:]
    targets = jax.lax.stop_gradient(
        rewards[:, :-1] + gammas[:, :-1] *
        jnp.max(
            (jax.nn.softmax(pi_beta_logits) > 1e-4).astype(jnp.float32) *  # Clipping based on pi_beta
            jnp.exp(jax.nn.log_softmax(target_logits) - jax.nn.log_softmax(pi_beta_logits)), axis=-1)
    )
    # Skip over target values computed over environment steps
    should_take_action = input_training_mask[:, 1:]
    masked_idxs = (
        should_take_action.astype(jnp.int32) * jnp.arange(0, should_take_action.shape[1])[None, ...] +
        jnp.flip(should_take_action, axis=1).astype(jnp.int32) * should_take_action.shape[1]
    )
    next_action_idxs = jax.lax.cummin(masked_idxs[:, ::-1], axis=-1)[:, ::-1]
    next_action_idxs = jnp.minimum(next_action_idxs, should_take_action.shape[1] - 1)
    targets = jnp.take_along_axis(targets, next_action_idxs, axis=1)

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

    mask = input_attention_mask[:, 1:] * input_training_mask[:, 1:]
    n = mask.sum()
    q_loss = softmax_cross_entropy(logits, target_distribution) * mask
    q_loss = q_loss.sum() / n
    bc_loss = softmax_cross_entropy_with_integer_labels(pi_beta_logits, target_ids) * mask
    bc_loss = bc_loss.sum() / n
    loss = q_loss + bc_loss

    logs = dict(
        losses=dict(
            total_loss=loss, 
            q_loss=q_loss,
            bc_loss=bc_loss
        ),
        targets=get_tensor_stats(targets, mask=mask, n=n), 
        rewards=get_tensor_stats(rewards, mask=mask, n=n), 
    )
    return loss, logs


def initialize_attn_mask_pos_ids(
    input_ids: jax.Array,  
    pad_token_id: Optional[Union[int, jax.Array]], 
    attention_mask: Optional[jax.Array]=None, 
    position_ids: Optional[jax.Array]=None, 
    position_id_shift: Optional[jax.Array]=None, 
) -> Tuple[jax.Array, jax.Array]:
    if attention_mask is None:
        if pad_token_id is None:
            attention_mask = jnp.ones_like(input_ids).astype(jnp.int32)
        else:
            attention_mask = (input_ids != pad_token_id).astype(jnp.int32)
    if position_ids is None:
        position_ids = jnp.maximum(jnp.cumsum(attention_mask, axis=1) - 1, 0).astype(jnp.int32)
        if position_id_shift is not None:
            position_ids = position_ids + position_id_shift[:, None]
    return attention_mask, position_ids

class GenerationFromStrOutput(NamedTuple):
    output_strs: List[str]
    scores: np.ndarray

# inference based on sequence-to-sequence input/outputs

class Inference(struct.PyTreeNode):
    params: PyTree
    target_params: Optional[PyTree]
    pi_beta_params: PyTree

    model: FlaxPreTrainedModel = struct.field(pytree_node=False)
    tokenizer: PreTrainedTokenizerBase = struct.field(pytree_node=False)
    _generate: Callable = struct.field(pytree_node=False)
    _forward: Callable = struct.field(pytree_node=False)
    _eval_loss: Optional[Callable] = struct.field(pytree_node=False, default=None)
    
    def generate(
        self, 
        input_ids: jax.Array, 
        prng_key: Optional[jax.random.PRNGKeyArray], 
        generation_config: Optional[GenerationConfig]=None, 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        trace: bool=True, 
    ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]:
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )

        return self._generate(
            self.params, 
            input_ids, 
            attention_mask, 
            position_ids, 
            prng_key, 
            freeze(generation_config.to_dict()) if generation_config is not None else None, 
            trace, 
        )
    
    def generate_from_str(
        self, 
        input_strs: List[str], 
        prng_key: Optional[jax.random.PRNGKeyArray], 
        blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.LEFT, truncation=Truncation.LEFT, max_length=None), 
        generation_config: Optional[GenerationConfig]=None, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        target_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        trace: bool=True, 
    ) -> GenerationFromStrOutput:
        if input_token_process is None:
            input_token_process = lambda x: x
        if target_token_process is None:
            target_token_process = lambda x: x
        # tokenize
        tokens = [input_token_process(self.tokenizer.encode(item)) for item in input_strs]
        tokens = block_sequences(tokens, self.tokenizer.pad_token_id, np.int32, blocking_strategy)
        # generate
        outputs = self.generate(
            jnp.asarray(tokens), 
            prng_key, 
            generation_config=generation_config, 
            trace=trace
        )
        # process outputs
        output_sequences = list(map(target_token_process, outputs.sequences.tolist()))
        output_scores = None
        if isinstance(outputs, FlaxBeamSearchOutput):
            output_scores = np.asarray(outputs.scores)
        # decode tokens
        output_strs = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        return GenerationFromStrOutput(output_strs, output_scores)
    
    def forward(
        self, 
        input_ids: jax.Array, 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        output_attentions: Optional[bool]=None, 
        output_hidden_states: Optional[bool]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    ) -> FlaxCausalLMOutput:
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )

        return self._forward(
            self.params, 
            input_ids, 
            attention_mask, 
            position_ids, 
            prng_key, 
            output_attentions, 
            output_hidden_states, 
            train, 
        )
    
    def forward_from_str(
        self, 
        input_strs: List[str], 
        blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.RIGHT, truncation=Truncation.RIGHT, max_length=None), 
        output_attentions: Optional[bool]=None, 
        output_hidden_states: Optional[bool]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> FlaxCausalLMOutput:
        if input_token_process is None:
            input_token_process = lambda x: x
        # tokenize
        tokens = [input_token_process(self.tokenizer.encode(item)) for item in input_strs]
        tokens = block_sequences(tokens, self.tokenizer.pad_token_id, np.int32, blocking_strategy)
        # forward
        outputs = self.forward(
            jnp.asarray(tokens), 
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, 
            train=train, 
            prng_key=prng_key, 
        )
        return outputs
    
    def eval_loss(
        self, 
        input_ids: jax.Array, 
        target_ids: jax.Array, 
        input_attention_mask: Optional[jax.Array]=None, 
        input_position_ids: Optional[jax.Array]=None, 
        target_attention_mask: Optional[jax.Array]=None, 
        target_position_ids: Optional[jax.Array]=None,
        rewards: Optional[jax.Array]=None,
        gammas: Optional[jax.Array]=None,
        prng_key: Optional[jax.random.PRNGKeyArray]=None,
        train: bool=False, 
    ) -> Tuple[jax.Array, PyTree]:
        input_attention_mask, input_position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            input_attention_mask, 
            input_position_ids, 
        )
        target_attention_mask, target_position_ids = initialize_attn_mask_pos_ids(
            target_ids, 
            self.tokenizer.pad_token_id, 
            target_attention_mask, 
            target_position_ids, 
            position_id_shift=input_position_ids.max(axis=1)+(input_attention_mask.sum(axis=1) > 0).astype(jnp.int32), 
        )

        return self._eval_loss(
            self.params,
            self.target_params,
            self.pi_beta_params,
            input_ids, 
            input_attention_mask, 
            input_position_ids, 
            target_ids, 
            target_attention_mask, 
            target_position_ids,
            rewards,
            gammas,
            prng_key, 
            train, 
        )


# train based on sequence-to-sequence input/outputs

class Train(struct.PyTreeNode):
    train_state: TrainState
    target_params: Optional[PyTree]
    pi_beta_train_state: TrainState
    model: FlaxPreTrainedModel = struct.field(pytree_node=False)
    tokenizer: PreTrainedTokenizerBase = struct.field(pytree_node=False)
    _step: Callable = struct.field(pytree_node=False)
    
    def step(
        self, 
        input_ids: jax.Array, 
        target_ids: jax.Array, 
        rewards: jax.Array,
        gammas: jax.Array,
        prng_key: Optional[jax.random.PRNGKeyArray], 
        input_attention_mask: Optional[jax.Array]=None, 
        input_position_ids: Optional[jax.Array]=None, 
        target_attention_mask: Optional[jax.Array]=None, 
        target_position_ids: Optional[jax.Array]=None,
        train: bool=True, 
    ) -> Tuple[Train, jax.Array, PyTree]:
        input_attention_mask, input_position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            input_attention_mask, 
            input_position_ids, 
        )
        target_attention_mask, target_position_ids = initialize_attn_mask_pos_ids(
            target_ids, 
            self.tokenizer.pad_token_id, 
            target_attention_mask, 
            target_position_ids, 
            position_id_shift=input_position_ids.max(axis=1)+(input_attention_mask.sum(axis=1) > 0).astype(jnp.int32), 
        )
        
        train_state, target_params, pi_beta_train_state, loss, logs = self._step(
            self.train_state,
            self.target_params,
            self.pi_beta_train_state,
            input_ids,
            input_attention_mask, 
            input_position_ids, 
            target_ids, 
            target_attention_mask, 
            target_position_ids,
            rewards,
            gammas,
            prng_key, 
            train, 
        )
        return self.replace(train_state=train_state,
                            target_params=target_params,
                            pi_beta_train_state=pi_beta_train_state), loss, logs

# inference based on inputs+binary mask

class InferenceMask(Inference):

    def eval_loss(
        self, 
        input_ids: jax.Array, 
        input_training_mask: jax.Array, 
        input_attention_mask: Optional[jax.Array]=None, 
        input_position_ids: Optional[jax.Array]=None,
        rewards: Optional[jax.Array]=None,
        gammas: Optional[jax.Array]=None,            
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        train: bool=False, 
    ) -> Tuple[jax.Array, PyTree]:
        input_attention_mask, input_position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            input_attention_mask, 
            input_position_ids, 
        )

        return self._eval_loss(
            self.params, 
            self.target_params,
            self.pi_beta_params,
            input_ids, 
            input_attention_mask, 
            input_position_ids, 
            input_training_mask,
            rewards,
            gammas,
            prng_key, 
            train, 
        )
    
    def eval_loss_from_str(self, *args, **kwargs) -> Tuple[jax.Array, PyTree]:
        raise NotImplementedError

# train based on inputs+binary mask

class TrainMask(Train):
    train_state: TrainState
    target_params: Optional[PyTree]
    pi_beta_train_state: TrainState
    model: FlaxPreTrainedModel = struct.field(pytree_node=False)
    tokenizer: PreTrainedTokenizerBase = struct.field(pytree_node=False)
    _step: Callable = struct.field(pytree_node=False)
    
    def step(
        self, 
        input_ids: jax.Array, 
        input_training_mask: jax.Array,
        rewards: jax.Array,
        gammas: jax.Array,   
        prng_key: Optional[jax.random.PRNGKeyArray], 
        input_attention_mask: Optional[jax.Array]=None, 
        input_position_ids: Optional[jax.Array]=None, 
        train: bool=True, 
    ) -> Tuple[Train, jax.Array, PyTree]:
        assert not jnp.any(input_training_mask[:, 0] > 0.0).item(), "input_training_mask[:, 0] should be all 0s, since cannot train on first token"

        input_attention_mask, input_position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            input_attention_mask, 
            input_position_ids, 
        )
        
        train_state, loss, logs = self._step(
            self.train_state,
            self.target_params,
            self.pi_beta_train_state,
            input_ids, 
            input_attention_mask, 
            input_position_ids, 
            input_training_mask,
            rewards,
            gammas,
            prng_key, 
            train, 
        )
        return self.replace(train_state=train_state), loss, logs
    
    def step_from_str(self, *args, **kwargs) -> Tuple[Train, jax.Array, PyTree]:
        raise NotImplementedError
