# TODO
# How to predict diffF
# predict 1-order tensor cartesian coordinate and transform to fraction?
# directly predict fraction (with/without sigmoid)
# use grad info like forces?
# How to predict diffC
# symmetry matrix?
# use strain mutation?
# use grad info like stress?
# use sum or mean?

from typing import Callable, List, Dict, Optional, Tuple
import numpy as np
import math
import torch
from torch import nn
from hotpp.layer.utils import ElementLinear
from hotpp.layer import EmbeddingLayer, RadialLayer
from hotpp.layer.equivalent import (
    NonLinearLayer,
    GraphConvLayer,
    AllegroGraphConvLayer,
    SelfInteractionLayer,
)
from hotpp.model.miao import MiaoBlock
from hotpp.utils import (
    expand_para,
    find_distances,
    _scatter_add,
    _scatter_mean,
    EnvPara,
)
from .base import CrystalDiffusion,CrystalEM
from .readout import ReadoutLayer

class EmoMiaoNet(CrystalEM):

    def __init__(
        self,
        cutoff: float,
        t_max: int,
        betas: List[float],
        sigmas: List[float],
        embedding_layer: EmbeddingLayer,
        time_embedding: nn.Module,
        radial_fn: RadialLayer,
        n_layers: int,
        max_r_way: List[int],
        max_out_way: List[int],
        output_dim: List[int],
        activate_fn: str = "silu",
        mean: float = 0.0,
        std: float = 1.0,
        norm_factor: float = 1.0,
    ):
        super().__init__(cutoff=cutoff, t_max=t_max, betas=betas, sigmas=sigmas)
        self.register_buffer("mean", torch.tensor(mean).float())
        self.register_buffer("std", torch.tensor(std).float())
        self.embedding_layer = embedding_layer
        self.time_embedding = time_embedding
        self.node_embedding = ElementLinear(
            EnvPara.ELEMENTS,
            input_dim=embedding_layer.n_channel + time_embedding.n_channel,
            output_dim=embedding_layer.n_channel,
        )
        self.edge_embedding = ElementLinear(
            EnvPara.ELEMENTS,
            input_dim=radial_fn.n_channel,
            output_dim=embedding_layer.n_channel,
        )
        self.radial_fn = radial_fn

        max_in_way = [0] + max_out_way[:-1]
        hidden_nodes = [embedding_layer.n_channel] + output_dim
        self.en_equivalent_blocks = self.get_eq_blocks(
            activate_fn,
            max_r_way,
            max_in_way,
            max_out_way,
            hidden_nodes,
            norm_factor,
            n_layers,
        )

        self.readout_layer = ReadoutLayer(
            n_dim=hidden_nodes[-1],
            target_way={"cart_velocity":1,"deform":2},
            target_channel={"cart_velocity":1,"deform":1},
            activate_fn=activate_fn,
        )

    def model_predict(
        self,
        batch_data: Dict[str, torch.Tensor],
        time: torch.Tensor,
        create_graph=True
    ) -> Dict[str, torch.Tensor]:
        node_info, edge_info = self.get_init_info(batch_data, time)
        for en_equivalent in self.en_equivalent_blocks:
            node_info, edge_info = en_equivalent(node_info, edge_info, batch_data)
        output_tensors = self.readout_layer(node_info, batch_data)
        #######################################################
        # # output_tensors["cart_diff"]: [n_atoms, 3]
        # # inv_cell: [n_atoms, 3, 3]
        # # output_tensors["cell_stress"]: [n_atoms, 3, 3]
        # # cell_stress: [n_batch, 3, 3]
        # # cell: [n_batch, 3, 3]
        # site_energy = output_tensors["site_energy_p"]
        # batch_data["energy_p"] = _scatter_add(site_energy, batch_data["batch"])
        # required_derivatives = ["coordinate","scaling"]
        # grads = torch.autograd.grad(
        #         [site_energy.sum()],
        #         [batch_data[prop] for prop in required_derivatives],
        #         create_graph=create_graph
        #     )
        # dE_dr = grads[required_derivatives.index("coordinate")]
        # dE_dl = grads[required_derivatives.index("scaling")]
        # batch_data["forces_p"] = -dE_dr
        # batch_data["virial_p"] = dE_dl
        output_tensors["cart_velocity"].squeeze()
        inv_cell = batch_data['inv_cell'].repeat_interleave(batch_data["n_atoms"], dim=0)
        frac_diff = torch.bmm(output_tensors["cart_velocity"].view(-1, 1, 3), inv_cell).squeeze(1)
        batch_data["frac_velocity"] = frac_diff
        deform_pa = output_tensors["deform"].squeeze()
        deform = _scatter_mean(deform_pa, batch_data["batch"])
        batch_data["deform"]=deform
        return batch_data


    def get_init_info(
        self,
        batch_data: Dict[str, torch.Tensor],
        time: torch.Tensor,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        symbol_emb = self.embedding_layer(batch_data=batch_data)
        time_emb = self.time_embedding(time).repeat_interleave(
            batch_data["n_atoms"], dim=0
        )
        emb = torch.cat((symbol_emb, time_emb), dim=1)
        node_info = {0: self.node_embedding(emb, batch_data)}
        _, dij, _ = find_distances(batch_data)
        edge_info = {0: self.edge_embedding(self.radial_fn(dij), batch_data)}
        return node_info, edge_info

    def get_eq_blocks(
        self,
        activate_fn,
        max_r_way,
        max_in_way,
        max_out_way,
        hidden_nodes,
        norm_factor,
        n_layers,
    ):
        return nn.ModuleList(
            [
                MiaoBlock(
                    activate_fn=activate_fn,
                    radial_fn=self.radial_fn.replicate(),
                    # Use factory method, so the radial_fn in each layer are different
                    max_r_way=max_r_way[i],
                    max_in_way=max_in_way[i],
                    max_out_way=max_out_way[i],
                    input_dim=hidden_nodes[i],
                    output_dim=hidden_nodes[i + 1],
                    norm_factor=norm_factor,
                )
                for i in range(n_layers)
            ]
        )