r"""
Naive gate
"""
from .base_gate import BaseGate

import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveGate(BaseGate):
    r"""
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indicies and the score, or confidence, are output to the parent
    module.
    The load-balance strategies are also designed to be implemented within the
    `Gate` module.
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2, gate_bias=True):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert, bias = gate_bias)
        self.top_k = top_k

    def forward(self, inp, return_all_scores=False):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        gate = self.gate(inp)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        
        if 'layer_count' not in globals():
            layer_count = 0
        if 'iteration_count' not in globals():
            iteration_count = 0
        
        if iteration_count % 100 == 0:
            output_file = f"/global/home/users/heatherhong/output/gate_output_{iteration_count}.csv"
            data = zip(gate_top_k_idx)
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                # 写入标题行
                writer.writerow([layer_count,iteration_count])
                # 写入数据行
                writer.writerows(data)
            print(f'Data has been written to {output_file}{layer_count}{iteration_count}.')
        
        if (layer_count + 1) % 24 == 0:
            layer_count = 0
            iteration_count += 1
        else:
            layer_count += 1


        # dummy loss
        self.set_loss(torch.zeros(1, requires_grad=True).cuda())

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score
