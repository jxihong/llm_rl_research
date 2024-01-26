from typing import Optional, Callable, Tuple
from jax.experimental.pjit import pjit
from LLM_RL.algorithms.ilql.base_interface import ILQLTrain, ILQLInference
from flax.training.train_state import TrainState
from jaxtyping import PyTree
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase
import flax.linen as nn
from JaxSeq.utils import with_named_sharding_constraint, match_partition_rules
from functools import partial
import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as PS
import jax.numpy as jnp
import optax
from LLM_RL.algorithms.value_rl_base.gpt2.interface import GPT2ValueRLInference

class GPT2ILQLTrain(ILQLTrain):
    @classmethod
    def load_train(
        cls, 
        base_train_state: TrainState, 
        target_base_params: Optional[PyTree],
        base_model: FlaxPreTrainedModel,
        tokenizer: PreTrainedTokenizerBase, 
        loss_fn: Callable, 
        detach: bool, 
    ):
        mesh = base_model.config.mesh
        assert mesh is not None
        base_train_state_partition_spec = match_partition_rules(base_model.config.get_partition_rules(), base_train_state)
        target_base_params_partition_spec = PS() if target_base_params is None else match_partition_rules(base_model.config.get_partition_rules(), target_base_params)

        @partial(
            pjit, 
            donate_argnums=(0, 1), 
            static_argnames=('train',), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), base_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), target_base_params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), base_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), target_base_params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
        )
        def _step(
            base_train_state: TrainState, 
            target_base_params: Optional[PyTree],

            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            position_ids: jax.Array, 
            should_take_action: jax.Array, 
            rewards: jax.Array, 
            dones: jax.Array, 

            next_token_ids: Optional[jax.Array], 
            next_tokens_attention_mask: Optional[jax.Array], 
            next_tokens_position_ids: Optional[jax.Array], 
            next_dones: Optional[jax.Array], 

            prng_key: Optional[jax.random.PRNGKeyArray], 
            train: bool=True, 
        ) -> Tuple[TrainState, Optional[PyTree], TrainState, TrainState, TrainState, PyTree, PyTree, jax.Array, PyTree]:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(('dp', 'fsdp'), None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS(('dp', 'fsdp'), None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS(('dp', 'fsdp'), None))
            should_take_action = with_named_sharding_constraint(should_take_action, mesh, PS(('dp', 'fsdp'), None))
            rewards = with_named_sharding_constraint(rewards, mesh, PS(('dp', 'fsdp'), None))
            dones = with_named_sharding_constraint(dones, mesh, PS(('dp', 'fsdp')))
            if next_token_ids is not None:
                assert next_tokens_attention_mask is not None
                assert next_tokens_position_ids is not None
                next_token_ids = with_named_sharding_constraint(next_token_ids, mesh, PS(('dp', 'fsdp'), None))
                next_tokens_attention_mask = with_named_sharding_constraint(next_tokens_attention_mask, mesh, PS(('dp', 'fsdp'), None))
                next_tokens_position_ids = with_named_sharding_constraint(next_tokens_position_ids, mesh, PS(('dp', 'fsdp'), None))
                next_dones = with_named_sharding_constraint(next_dones, mesh, PS(('dp', 'fsdp')))
            else:
                assert next_tokens_attention_mask is None
                assert next_tokens_position_ids is None

            # define loss function

            def grad_loss(base_params: PyTree, prng_key: jax.random.PRNGKeyArray:
                
                # get base hidden states

                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                base_model_output = base_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids, 
                    params=base_params, 
                    dropout_rng=new_key, 
                    train=train, 
                    output_hidden_states=False, 
                )

                if target_base_params is not None:
                    new_key = None
                    if prng_key is not None:
                        prng_key, new_key = jax.random.split(prng_key)
                    target_base_model_output = base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask, 
                        position_ids=position_ids, 
                        params=target_base_params, 
                        dropout_rng=new_key, 
                        train=train, 
                        output_hidden_states=False, 
                    )
                else:
                    target_base_model_output = base_model_output
                
                if next_token_ids is not None:
                    new_key = None
                    if prng_key is not None:
                        prng_key, new_key = jax.random.split(prng_key)
                    next_token_base_model_output = base_model(
                        input_ids=next_token_ids, 
                        attention_mask=next_tokens_attention_mask, 
                        position_ids=next_tokens_position_ids, 
                        params=base_params, 
                        dropout_rng=new_key, 
                        train=train, 
                        output_hidden_states=True, 
                    )
                
                # get values
                p_logits = base_model_output.logits
                target_p_logits = target_base_model_output.logits

                # stop gradients
                if detach:
                    p_logits = jax.lax.stop_gradient(p_logits)
                target_p_logits = jax.lax.stop_gradient(target_p_logits)

                p = jnp.take_along_axis(jax.nn.softmax(p_logits[:, :-1]), input_ids[:, 1:][..., None], axis=2).squeeze(2)
                target_p = jnp.take_along_axis(jax.nn.softmax(target_p_logits[:, :-1]), input_ids[:, 1:][..., None], axis=2).squeeze(2)
                # get next token values
                if next_token_ids is not None:
                    # just run vf on last token to save some flops
                    last_next_token_idxs = (next_tokens_attention_mask.shape[1]-1)-jnp.argmax(jnp.flip(next_tokens_attention_mask, axis=1).astype(jnp.int32), axis=1)
                    final_p = next_token_base_model_output.logits[jnp.arange(0, input_ids.shape[0], dtype=jnp.int32), last_next_token_idxs, :]
                    final_p = final_p * (1 - next_dones.astype(jnp.float32))
                else:
                    last_action_idxs = (should_take_action.shape[1]-1)-jnp.argmax(jnp.flip(should_take_action, axis=1).astype(jnp.int32), axis=1)+1
                    last_token_idxs = (attention_mask.shape[1]-1)-jnp.argmax(jnp.flip(attention_mask, axis=1).astype(jnp.int32), axis=1)
                    final_state_idxs = ((1 - dones) * last_action_idxs + dones * last_token_idxs).astype(jnp.int32)
                    final_p = target_p_logits[jnp.arange(0, should_take_action.shape[0], dtype=jnp.int32), final_state_idxs]
                    final_p = final_p * (1 - dones)
                final_p = jax.lax.stop_gradient(final_p)

                loss, info = loss_fn(
                    p,
                    target_p,
                    final_p,
                    p_logits, 
                    input_ids[:, 1:], 
                    attention_mask[:, 1:], 
                    should_take_action, 
                    rewards, 
                )
                return loss, info

            # take loss
            (loss, info), (base_grads,) = jax.value_and_grad(grad_loss, has_aux=True, argnums=(0,))(
                base_train_state.params, 
                prng_key, 
            )
            # assert shard gradients
            base_grads = jax.tree_util.tree_map(
                lambda x, ps: with_named_sharding_constraint(x, mesh, ps), 
                base_grads, 
                base_train_state_partition_spec.params, 
            )
            # update params and optim state
            base_train_state = base_train_state.apply_gradients(grads=base_grads)

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
            
            target_base_params = jax.lax.cond(
                update_cond(base_train_state.opt_state), 
                update_targets, 
                mid_targets, 
                base_train_state.params, 
                target_base_params, 
                base_train_state.step, 
            )

            return base_train_state, target_base_params, loss, info

        return cls(
            base_train_state=base_train_state, 
            target_base_params=target_base_params, 
            base_model=base_model, 
            tokenizer=tokenizer, 
            _step=_step, 
        )


class GPT2ILQLInference(ILQLInference):
    @classmethod
    def load_inference(
        cls, 
        value_inference: GPT2ValueRLInference, 
        target_value_inference: GPT2ValueRLInference, 
        loss_fn: Callable, 
        use_target_base_for_loss: bool=True, 
    ):
        mesh = value_inference.base_model.config.mesh
        assert mesh is not None
        assert mesh == target_value_inference.base_model.config.mesh

        base_params_partition_spec = match_partition_rules(value_inference.base_model.config.get_partition_rules(), value_inference.base_params)
        target_base_params_partition_spec = PS() if (not use_target_base_for_loss) else match_partition_rules(target_value_inference.base_model.config.get_partition_rules(), target_value_inference.base_params)
        
        @partial(
            pjit, 
            static_argnames=('train',), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), base_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), target_base_params_partition_spec), 
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
            base_params: PyTree, 
            target_base_params: Optional[PyTree], 

            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            position_ids: jax.Array, 
            should_take_action: jax.Array, 
            rewards: jax.Array, 
            dones: jax.Array, 

            next_token_ids: Optional[jax.Array], 
            next_tokens_attention_mask: Optional[jax.Array], 
            next_tokens_position_ids: Optional[jax.Array], 
            next_dones: Optional[jax.Array], 

            prng_key: Optional[jax.random.PRNGKeyArray]=None, 
            train: bool=False, 
        ) -> Tuple[jax.Array, PyTree]:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(('dp', 'fsdp'), None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS(('dp', 'fsdp'), None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS(('dp', 'fsdp'), None))
            should_take_action = with_named_sharding_constraint(should_take_action, mesh, PS(('dp', 'fsdp'), None))
            rewards = with_named_sharding_constraint(rewards, mesh, PS(('dp', 'fsdp'), None))
            dones = with_named_sharding_constraint(dones, mesh, PS(('dp', 'fsdp')))
            if next_token_ids is not None:
                assert next_tokens_attention_mask is not None
                assert next_tokens_position_ids is not None
                next_token_ids = with_named_sharding_constraint(next_token_ids, mesh, PS(('dp', 'fsdp'), None))
                next_tokens_attention_mask = with_named_sharding_constraint(next_tokens_attention_mask, mesh, PS(('dp', 'fsdp'), None))
                next_tokens_position_ids = with_named_sharding_constraint(next_tokens_position_ids, mesh, PS(('dp', 'fsdp'), None))
                next_dones = with_named_sharding_constraint(next_dones, mesh, PS(('dp', 'fsdp')))
            else:
                assert next_tokens_attention_mask is None
                assert next_tokens_position_ids is None
                
            # get base hidden states

            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            base_model_output = value_inference.base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                params=base_params, 
                dropout_rng=new_key, 
                train=train, 
                output_hidden_states=True, 
            )

            if target_base_params is not None:
                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                target_base_model_output = target_value_inference.base_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids, 
                    params=target_base_params, 
                    dropout_rng=new_key, 
                    train=train, 
                    output_hidden_states=True, 
                )
            else:
                target_base_model_output = base_model_output
            
            if next_token_ids is not None:
                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                next_token_base_model_output = value_inference.base_model(
                    input_ids=next_token_ids, 
                    attention_mask=next_tokens_attention_mask, 
                    position_ids=next_tokens_position_ids, 
                    params=base_params, 
                    dropout_rng=new_key, 
                    train=train, 
                    output_hidden_states=True, 
                )
            
            # process outputs

            p_logits = base_model_output.logits[:, :-1]
            target_p_logits = target_base_model_output.logits[:, :-1]

            # get next token values
            if next_token_ids is not None:
                # just run vf on last token to save some flops
                last_next_token_idxs = (next_tokens_attention_mask.shape[1]-1)-jnp.argmax(jnp.flip(next_tokens_attention_mask, axis=1).astype(jnp.int32), axis=1)
                final_p_logits = next_token_base_model_output.logits[jnp.arange(0, input_ids.shape[0], dtype=jnp.int32), last_next_token_idxs, :]
                final_p_logits = final_logits * (1 - next_dones.astype(jnp.float32))
            else:
                last_action_idxs = (should_take_action.shape[1]-1)-jnp.argmax(jnp.flip(should_take_action, axis=1).astype(jnp.int32), axis=1)+1
                last_token_idxs = (attention_mask.shape[1]-1)-jnp.argmax(jnp.flip(attention_mask, axis=1).astype(jnp.int32), axis=1)
                final_state_idxs = ((1 - dones) * last_action_idxs + dones * last_token_idxs).astype(jnp.int32)
                final_p_logits = target_p_logits[jnp.arange(0, should_take_action.shape[0], dtype=jnp.int32), final_state_idxs]
                final_p_logits = final_logits * (1 - dones)

            loss, info = loss_fn(
                p_logits, 
                target_p_logits, 
                final_p_logits, 
                input_ids[:, 1:], 
                attention_mask[:, 1:], 
                should_take_action, 
                rewards, 
            )
            
            return loss, info

        return cls(
            value_inference=value_inference, 
            target_value_inference=target_value_inference, 
            _eval_loss=_eval_loss, 
            use_target_base_for_loss=use_target_base_for_loss, 
        )
