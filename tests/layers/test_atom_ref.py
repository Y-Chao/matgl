from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from matgl.layers._atom_ref import AtomRef


def _batch(graphs):
    return Batch.from_data_list(graphs)


def test_atom_ref(graph_MoSH):
    _, g1, _ = graph_MoSH
    element_ref = AtomRef(torch.tensor([0.5, 1.0, 2.0]))
    assert element_ref(g1) == 3.5


def test_atom_ref_without_property_offset(graph_MoSH):
    _, g1, _ = graph_MoSH
    element_ref = AtomRef()
    assert element_ref(g1) == 0.0


def test_atom_ref_property_offset_as_list(graph_MoSH):
    _, g1, _ = graph_MoSH
    element_ref = AtomRef([0.5, 1.0, 2.0])
    assert element_ref(g1) == 3.5


def test_atom_ref_fit(graph_MoSH):
    _, g1, _ = graph_MoSH
    element_ref = AtomRef(torch.tensor([0.5, 1.0, 2.0]))
    properties = torch.tensor([2.0, 2.0])
    bg = _batch([g1, g1])
    element_ref.fit([g1, g1], properties)

    atom_ref = element_ref(bg)
    assert list(np.round(atom_ref.numpy())) == [2.0, 2.0]


def test_atom_ref_with_states(graph_MoSH):
    _, g1, _ = graph_MoSH
    element_ref = AtomRef(torch.tensor([[0.5, 1.0, 2.0], [2.0, 3.0, 5.0]]))
    state_label = torch.tensor([1])
    assert element_ref(g1, state_label) == 10


def test_atom_ref_with_states_batched(graph_MoSH):
    """Multi-state refs pick one row per graph for a mixed-state batch (B,), not a (B, B) cross product."""
    _, g1, _ = graph_MoSH
    element_ref = AtomRef(torch.tensor([[0.5, 1.0, 2.0], [2.0, 3.0, 5.0]]))
    single = [element_ref(g1, torch.tensor([s])) for s in (0, 1)]
    out = element_ref(_batch([g1, g1, g1]), torch.tensor([1, 0, 1]))
    assert out.shape == (3,)
    assert torch.allclose(out, torch.stack([single[1], single[0], single[1]]).reshape(-1))


def test_atom_ref_with_states_requires_one_label_per_graph(graph_MoSH):
    _, g1, _ = graph_MoSH
    element_ref = AtomRef(torch.tensor([[0.5, 1.0, 2.0], [2.0, 3.0, 5.0]]))
    for bad in (None, torch.tensor([0.0, 0.0])):
        with pytest.raises(ValueError, match="state_attr"):
            element_ref(g1, bad)
