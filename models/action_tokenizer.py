import pickle
from typing import List, Union, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

def transform_to_global(
    pos_local: Tensor,  # [n_agent, n_step, 2]
    head_local: Optional[Tensor],  # [n_agent, n_step]
    pos_now: Tensor,  # [n_agent, 2]
    head_now: Tensor,  # [n_agent]
) -> Tuple[Tensor, Optional[Tensor]]:
    cos, sin = head_now.cos(), head_now.sin()
    rot_mat = torch.zeros((head_now.shape[0], 2, 2), device=head_now.device)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = sin
    rot_mat[:, 1, 0] = -sin
    rot_mat[:, 1, 1] = cos

    pos_global = torch.bmm(pos_local, rot_mat)  # [n_agent, n_step, 2]*[n_agent, 2, 2]
    pos_global = pos_global + pos_now.unsqueeze(1)
    if head_local is None:
        head_global = None
    else:
        head_global = head_local + head_now.unsqueeze(1)
    return pos_global, head_global


class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, model_config: dict) -> None:
        """
        Discretizes continuous vehicle actions into N bins per dimension and maps to the least used tokens.

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param model_config: model configuration as dictionary.
        """
        self.action_start_id = model_config['tokens']['action_start_id']
        codebook_path = model_config['codebook_cache_path']
        with open(codebook_path, "rb") as f:
            code_book = pickle.load(f)['token_all']['veh'] 
            self.code_book = torch.tensor(code_book)  # (n_bins, 6, 4, 2)
        
        action_len = self.code_book.shape[0]
        # Add action tokens to tokenizer
        tokenizer.add_tokens([f'<action_{i}>' for i in range(action_len)], special_tokens=False)
        
        # Store tokenizer
        self.tokenizer = tokenizer
        self.n_bins = action_len

    def __call__(self, action_token: np.ndarray) -> Union[str, List[str]]:
        # convert to text for tokenization
        action = ''
        for i in range(action_token.shape[0]):
            action += f'<action_{action_token[i]}>'

        return action
        
    def decode_token_ids_to_trajectory(self, token_ids: torch.Tensor) -> np.ndarray:
        """
        Returns continuous states (trajectory) from token IDs.
        """
        # decode token ids to action
        action_token_ids = []

        for i in range(len(token_ids)):
            if token_ids[i] < self.action_start_id: # not valid action token
                action_token_ids.append(0)
            else:
                action = self.tokenizer.decode(token_ids[i])
                action_token_ids.append(int(action.split('_')[1].replace('>', '')))
        
        # print(action_token_ids)
        action_token_ids = torch.tensor(action_token_ids, dtype=torch.long)

        try:
            action_tokens = self.code_book[action_token_ids]
        except Exception as e:
            print(f"Error type: {type(e).__name__}")
            print(action_token_ids)
            print(f"Error message: {str(e)}")
            return torch.zeros(1, 1, 3)

        time_steps = action_tokens.shape[0]
        traj = self.rollout(action_tokens, time_steps=time_steps)
        
        return traj

    def rollout(self, action_tokens: torch.Tensor, time_steps: int) -> torch.Tensor:
        # initial state
        pos_a = torch.tensor([[[0, 0]]]) # [1, 1, 2]
        head_a = torch.tensor([[0]]) # [1, 1]
        
        # loop through all tokens
        for t in range(time_steps):
            next_token_traj_all = action_tokens[None, t]  # [1, 6, 4, 2]
            
            # transform to global
            token_traj_global = transform_to_global(
                pos_local=next_token_traj_all.flatten(1, 2),  # [1, 6*4, 2]
                head_local=None,
                pos_now=pos_a[:, t],  # [1, 2]
                head_now=head_a[:, t],  # [1]
            )[0].view(*next_token_traj_all.shape)

            # get pos_a_next and head_a_next
            pos_a_next = token_traj_global[:, -1].mean(dim=1)
            diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]
            head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])
            
            # get trajectory
            pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
            head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)

        # output trajectory
        trajectory = torch.cat([pos_a, head_a.unsqueeze(-1)], dim=-1)

        return trajectory

    @property
    def vocab_size(self) -> int:
        return self.n_bins