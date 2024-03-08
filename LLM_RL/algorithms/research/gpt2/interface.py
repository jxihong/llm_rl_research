from __future__ import annotations
import jax
import jax.numpy as jnp
from JaxSeq.stream_tokens import StreamingGenerationConfig
from jax.sharding import PartitionSpec as PS
from jaxtyping import PyTree
from functools import partial
from typing import Optional, Union, Tuple, Callable
import optax
from transformers.tokenization_utils import PreTrainedTokenizerBase
from JaxSeq.utils import with_named_sharding_constraint, match_partition_rules
from optax import softmax_cross_entropy_with_integer_labels
from flax.training.train_state import TrainState
from transformers.modeling_flax_outputs import FlaxCausalLMOutput, FlaxCausalLMOutputWithCrossAttentions
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.generation import FlaxBeamSearchOutput, FlaxGreedySearchOutput, FlaxSampleOutput
from flax.core import FrozenDict
from jax.sharding import NamedSharding
from jax.experimental.pjit import pjit
from LLM_RL.algorithms.research.base_interface import loss_fn_mask, TrainMask, InferenceMask


class GPT2TrainMask(TrainMask):
    @classmethod
    def load_train(
        cls,
        train_state: TrainState,
        polyak_alpha: float, 
        target_params: Optional[PyTree],
        pi_beta_train_state: TrainState,
        model: FlaxPreTrainedModel, 
        tokenizer: PreTrainedTokenizerBase, 
        loss_fn: Callable=loss_fn_mask,
        hard_update_every: Optional[int]=10, 
    ) -> GPT2TrainMask:
        mesh = model.config.mesh
        assert mesh is not None
        train_state_partition_spec = match_partition_rules(model.config.get_partition_rules(), train_state)
        target_params_partition_spec = PS() if target_params is None else match_partition_rules(model.config.get_partition_rules(), target_params)
        pi_beta_train_state_partition_spec = match_partition_rules(model.config.get_partition_rules(), pi_beta_train_state.params)
        
        @partial(
            pjit, 
            donate_argnums=(0, 1, 2), 
            static_argnames=('train',), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), target_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), pi_beta_train_state_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), target_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), pi_beta_train_state_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
        )
        def _step(
            train_state: TrainState,
            target_params: Optional[PyTree],
            pi_beta_params: PyTree,
            pi_beta_train_state: TrainState,
            input_ids: jax.Array, 
            input_attention_mask: jax.Array, 
            input_position_ids: jax.Array, 
            input_training_mask: jax.Array,
            rewards: jax.Array,
            gammas: jax.Array,
            prng_key: Optional[jax.random.PRNGKeyArray], 
            train: bool=True, 
        ) -> Tuple[TrainState, jax.Array, PyTree]:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(("dp", "fsdp"), None))
            input_attention_mask = with_named_sharding_constraint(input_attention_mask, mesh, PS(("dp", "fsdp"), None))
            input_position_ids = with_named_sharding_constraint(input_position_ids, mesh, PS(("dp", "fsdp"), None))
            input_training_mask = with_named_sharding_constraint(input_training_mask, mesh, PS(("dp", "fsdp"), None))
            rewards = with_named_sharding_constraint(rewards, mesh, PS(('dp', 'fsdp'), None))
            gammas = with_named_sharding_constraint(gammas, mesh, PS(('dp', 'fsdp'), None))

            # define loss function
            def grad_loss(params: PyTree):
                loss, info = loss_fn(
                    model, 
                    params,
                    target_params,
                    pi_beta_params,
                    input_ids, 
                    input_attention_mask, 
                    input_position_ids, 
                    input_training_mask,
                    rewards,
                    gammas,
                    prng_key, 
                    train, 
                )
                return loss, info

            # take loss
            (loss, info), (grads, pi_beta_grads) = jax.value_and_grad(grad_loss, has_aux=True)(
                train_state.params,
                pi_beta_train_state.params,
                prng_key
            )
            # assert shard gradients
            grads = jax.tree_util.tree_map(
                lambda x, ps: with_named_sharding_constraint(x, mesh, ps), grads,
                train_state_partition_spec.params)
            pi_beta_grads = jax.tree_util.tree_map(
                lambda x, ps: with_named_sharding_constraint(x, mesh, ps), pi_beta_grads,
                pi_beta_train_state_partition_spec.params)
            # update params and optim state
            train_state = train_state.apply_gradients(grads=grads)
            pi_beta_train_state = pi_beta_train_state.apply_gradients(grads=pi_beta_grads)

            # handle target network updates
            def update_targets(params: PyTree, base_params: PyTree, steps: jnp.ndarray) -> PyTree:
                base_params = optax.incremental_update(params, base_params, polyak_alpha)
                if hard_update_every is not None:
                    base_params = optax.periodic_update(params, base_params, steps, hard_update_every)
                return base_params
            
            def mid_targets(params: PyTree, base_params: PyTree, steps: jnp.ndarray) -> PyTree:
                return base_params

            def update_cond(opt_state: PyTree) -> bool:
                if hasattr(opt_state, 'mini_step'):
                    return opt_state.mini_step == 0
                return True
            
            if target_params is not None:
                target_params = jax.lax.cond(
                    update_cond(train_state.opt_state), 
                    update_targets, 
                    mid_targets, 
                    train_state.params, 
                    target_params, 
                    train_state.step, 
                )
            return train_state, target_params, pi_beta_params, loss, info
        
        return cls(
            train_state=train_state,
            target_params=target_params,
            pi_beta_train_state=pi_beta_train_state,
            model=model, 
            tokenizer=tokenizer, 
            _step=_step, 
        )


class GPT2InferenceMask(InferenceMask):
    @classmethod
    def load_inference(
        cls, 
        params: PyTree,
        target_params: Optional[PyTree],
        pi_beta_params: PyTree,
        model: FlaxPreTrainedModel, 
        tokenizer: PreTrainedTokenizerBase, 
        loss_fn: Optional[Callable]=loss_fn_mask, 
        dp_shard_logits: bool=True, 
    ) -> GPT2InferenceMask:
        mesh = model.config.mesh
        assert mesh is not None
        params_partition_spec = match_partition_rules(model.config.get_partition_rules(), params)
        target_params_partition_spec = PS() if target_params is None else match_partition_rules(model.config.get_partition_rules(), target_params)
        pi_beta_params_partition_spec = match_partition_rules(model.config.get_partition_rules(), pi_beta_params)


        @partial(
            pjit, 
            static_argnames=('generation_config', 'trace'), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=NamedSharding(mesh, PS()), 
        )
        def _generate(
            params: PyTree, 
            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            position_ids: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray], 
            generation_config: Optional[FrozenDict]=None, 
            trace: bool=True, 
        ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(("dp", "fsdp"), None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS(("dp", "fsdp"), None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS(("dp", "fsdp"), None))
            # NOTE: position_ids ignored by transformers

            # generate from model
            output = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                params=params, 
                prng_key=prng_key, 
                generation_config=StreamingGenerationConfig.from_dict(generation_config) if generation_config is not None else None, 
                trace=trace, 
            )
            
            return output
    
        @partial(
            pjit, 
            static_argnames=('output_attentions', 'output_hidden_states', 'train'), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=FlaxCausalLMOutputWithCrossAttentions(
                logits=NamedSharding(mesh, PS(("dp", "fsdp"), None, None)) if dp_shard_logits else NamedSharding(mesh, PS()), 
                past_key_values=NamedSharding(mesh, PS()), # assume no sharding for past key values
                hidden_states=NamedSharding(mesh, PS()), # assume no sharding for hidden states
                attentions=NamedSharding(mesh, PS()), # assume no sharding for attentions
                cross_attentions=NamedSharding(mesh, PS()), # assume no sharding for cross attentions
            )
        )
        def _forward(
            params: PyTree, 
            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            position_ids: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray]=None, 
            output_attentions: Optional[bool]=None, 
            output_hidden_states: Optional[bool]=None, 
            train: bool=False, 
        ) -> FlaxCausalLMOutputWithCrossAttentions:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(("dp", "fsdp"), None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS(("dp", "fsdp"), None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS(("dp", "fsdp"), None))

            # get logits
            output = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                params=params, 
                train=train, 
                output_attentions=output_attentions, 
                output_hidden_states=output_hidden_states, 
                dropout_rng=prng_key, 
            )
            # trunc padded logits
            output = output.replace(logits=output.logits.at[:, :, model.config.unpadded_vocab_size:].set(-float('inf')))

            # assert sharding on outputs
            if dp_shard_logits:
                output = output.replace(logits=with_named_sharding_constraint(output.logits, mesh, PS(("dp", "fsdp"), None, None)))
            return output

        @partial(
            pjit, 
            static_argnames=('train',), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), target_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), pi_beta_params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=(
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
        )
        def _eval_loss(
            params: PyTree,
            target_params: Optional[PyTree],
            pi_beta_params: PyTree,
            input_ids: jax.Array, 
            input_attention_mask: jax.Array, 
            input_position_ids: jax.Array, 
            input_training_mask: jax.Array,
            rewards: jax.Array,
            gammas: jax.Array,
            prng_key: Optional[jax.random.PRNGKeyArray], 
            train: bool=False, 
        ) -> Tuple[jax.Array, PyTree]:
            assert loss_fn is not None, "loss_fn must be set to use eval_loss"
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(("dp", "fsdp"), None))
            input_attention_mask = with_named_sharding_constraint(input_attention_mask, mesh, PS(("dp", "fsdp"), None))
            input_position_ids = with_named_sharding_constraint(input_position_ids, mesh, PS(("dp", "fsdp"), None))
            input_training_mask = with_named_sharding_constraint(input_training_mask, mesh, PS(("dp", "fsdp"), None))
            rewards = with_named_sharding_constraint(rewards, mesh, PS(('dp', 'fsdp'), None))
            gammas = with_named_sharding_constraint(gammas, mesh, PS(('dp', 'fsdp'), None))

            # define loss function
            loss, info = loss_fn(
                model, 
                params,
                target_params,
                pi_beta_params,
                input_ids, 
                input_attention_mask, 
                input_position_ids, 
                input_training_mask,
                rewards,
                gammas,
                prng_key, 
                train, 
            )
            return loss, info
        
        return cls(
            params=params,
            target_params=target_params,
            pi_beta_params=pi_beta_params,
            model=model, 
            tokenizer=tokenizer, 
            _generate=_generate, 
            _forward=_forward, 
            _eval_loss=_eval_loss, 
        )
