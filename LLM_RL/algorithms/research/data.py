from __future__ import annotations
from typing import Dict, Iterable, List, Iterator, NamedTuple, Optional
from JaxSeq.utils import Dataset, IterableDataset, block_sequences, BlockingStrategy
import numpy as np
import jax.numpy as jnp
import jax
from transformers.tokenization_utils import PreTrainedTokenizerBase
from LLM_RL.environment import TokenTrajectoryChain

class RLData(NamedTuple):
    input_ids: np.ndarray # [t]
    input_training_mask: np.ndarray # [t]  whether should take action
    rewards: np.ndarray # [t]
    gammas: np.ndarray # [t]

    @staticmethod
    def block(
        data: List[RLData], 
        blocking_strategy: BlockingStrategy, 
        tokenizer: PreTrainedTokenizerBase, 
    ) -> Dict[str, np.ndarray]:
        return dict(
            input_ids=block_sequences(
                list(map(lambda x: x.input_ids, data)), 
                tokenizer.pad_token_id, 
                dtype=np.int32, 
                blocking_strategy=blocking_strategy, 
            ), 
            input_training_mask=block_sequences(
                list(map(lambda x: x.input_training_mask, data)), 
                False, 
                dtype=np.bool_, 
                blocking_strategy=blocking_strategy._replace(max_length=blocking_strategy.max_length), 
            ), 
            rewards=block_sequences(
                list(map(lambda x: x.rewards, data)), 
                0.0, 
                dtype=np.float32, 
                blocking_strategy=blocking_strategy._replace(max_length=blocking_strategy.max_length), 
            ),
            gammas=block_sequences(
                list(map(lambda x: x.gammas, data)), 
                1.0, 
                dtype=np.float32, 
                blocking_strategy=blocking_strategy._replace(max_length=blocking_strategy.max_length), 
            ), 

        )
    
    @classmethod
    def from_token_trajectory_chain(
        cls, 
        token_trajectory_chain: TokenTrajectoryChain,
        gamma: float, 
    ):
        gammas = gamma * np.ones_like(token_trajectory_chain.tokens, dtype=np.float32)
        # do not discount in between utterances
        gammas[1:][token_trajectory_chain.is_action[1:] <= token_trajectory_chain.is_action[:-1]] = 1.0

        return cls(
            input_ids=token_trajectory_chain.token_trajectory.tokens, 
            input_training_mask=token_trajectory_chain.token_trajectory.is_action, 
            rewards=token_trajectory_chain.token_trajectory.reward,
            gammas=gammas
        )


class RLDataset(Dataset):
    def __init__(
        self, 
        input_ids: np.ndarray, # [b, t]
        input_training_mask: np.ndarray, # [b, t]
        rewards: np.ndarray, # [b, t]
        gammas: np.ndarray, #[b, t]
    ):
        assert input_ids.shape == input_training_mask.shape
        assert input_ids.shape == rewards.shape
        assert input_ids.shape == gammas.shape

        self.input_ids = input_ids
        self.input_training_mask = input_training_mask
        self.rewards = rewards
        self.gammas = gammas
    
    def __getitem__(self, index):
        return {
            'input_ids': jnp.asarray(self.input_ids[index], dtype=jnp.int32), 
            'input_training_mask': jnp.asarray(self.input_training_mask[index], dtype=jnp.bool_), 
            'rewards': jnp.asarray(self.rewards[index], dtype=jnp.float32),
            'gammas': jnp.asarray(self.gammas[index], dtype=jnp.float32)
        }
    
    def __len__(self):
        return self.input_ids.shape[0]
    
    @classmethod
    def from_ilql_data_list(
        cls, 
        rl_data_list: List[RLData], 
        tokenizer: PreTrainedTokenizerBase, 
        blocking_strategy: BlockingStrategy, 
    ) -> RLDataset:
        
        data = RLData.block(rl_data_list, blocking_strategy, tokenizer)

        return cls(**data)


class _RLIteratorDataset:
    def __init__(self, rl_data: Iterator[Dict[str, np.ndarray]]):
        self.rl_data = rl_data

    def __next__(self):
        item = next(self.rl_data)
        return {
            'input_ids': jnp.asarray(item['input_ids'], dtype=jnp.int32), 
            'input_training_mask': jnp.asarray(item['input_training_mask'], dtype=jnp.bool_), 
            'rewards': jnp.asarray(item['rewards'], dtype=jnp.float32),
            'gammas': jnp.asarray(item['gammas'], dtype=jnp.float32),
        }

class RLIterableDataset(IterableDataset):
    def __init__(self, rl_data: Iterable[Dict[str, np.ndarray]]):
        self.rl_data = rl_data
    
    def __iter__(self):
        return _RLIteratorDataset(iter(self.rl_data))
    
    @classmethod
    def from_ilql_data_iterable(
        cls, 
        rl_data: Iterable[RLData], 
        tokenizer: PreTrainedTokenizerBase, 
        blocking_strategy: BlockingStrategy, 
    ) -> RLIterableDataset:
        
        class _TokensIterable(Iterable):
            def _tokens_generator(self):
                for item in rl_data:
                    yield jax.tree_util.tree_map(lambda x: x[0], RLData.block([item], blocking_strategy, tokenizer))

            def __iter__(self):
                return self._tokens_generator()

        return cls(_TokensIterable())
