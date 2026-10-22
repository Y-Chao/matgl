"""Multi-fidelity (state-conditioned) TensorNet/QET on the warp kernels.

Every test builds a PyG model, randomizes its state parameters (the FiLM scales start at zero, which would hide
state bugs), copies the weights into a warp model and checks that both backends agree. Requires ``nvalchemiops``
(the warp kernels also run on CPU, e.g. ``uv run --with warp-lang python -m pytest tests/models/test_mfi_warp.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch_geometric.data import Batch

from matgl.apps.pes import Potential
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph._compute import compute_pair_vector_and_distance
from matgl.models import QET
from matgl.models._tensornet import TensorNet, _warp_available
from matgl.utils.maths import tensor_norm

if TYPE_CHECKING:
    from pymatgen.core import Structure

pytestmark = pytest.mark.skipif(not _warp_available, reason="Warp tests require nvalchemiops to be installed.")

ELEMENTS = ("Mo", "S")
MFI_KWARGS = {"include_state": True, "ntypes_state": 2, "dim_state_embedding": 8}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _randomize_state_params(model: torch.nn.Module) -> None:
    torch.manual_seed(1)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "state" in name:
                param.normal_(std=0.1)


def _pyg_warp_pair(model_cls=TensorNet, **kwargs):
    torch.manual_seed(0)
    model_pyg = model_cls(element_types=ELEMENTS, use_warp=False, **MFI_KWARGS, **kwargs)
    _randomize_state_params(model_pyg)
    model_warp = model_cls(element_types=ELEMENTS, use_warp=True, **MFI_KWARGS, **kwargs)
    model_warp.load_state_dict(model_pyg.state_dict())
    assert model_warp._use_warp
    assert not model_pyg._use_warp
    return model_pyg.to(DEVICE), model_warp.to(DEVICE)


@pytest.fixture(scope="module")
def structure(MoS) -> Structure:
    """Distorted 2x2x2 MoS supercell: non-zero forces and more than one environment per element."""
    s = MoS.copy()
    s.make_supercell([2, 2, 2])
    s.perturb(0.1, min_distance=0.05, seed=0)
    return s


def _graph(structure: Structure):
    graph, lattice, _ = Structure2Graph(element_types=ELEMENTS, cutoff=5.0).get_graph(structure)
    return graph, lattice


def _batch(structure: Structure, n: int = 2):
    """A batch of ``n`` copies of ``structure`` with positions set, plus the stacked lattices."""
    graph, lattice = _graph(structure)
    batch = Batch.from_data_list([graph.clone() for _ in range(n)]).to(DEVICE)
    lattices = torch.cat([lattice] * n).to(DEVICE)
    lat_per_node = lattices[batch.batch]
    batch.pos = torch.einsum("ni,nij->nj", batch.frac_coords, lat_per_node)
    batch.pbc_offshift = torch.einsum("ei,eij->ej", batch.pbc_offset, lattices[batch.batch[batch.edge_index[0]]])
    return batch, lattices


def _layer_inputs(model: TensorNet, g):
    bond_vec, bond_dist = compute_pair_vector_and_distance(g.pos, g.edge_index, g.pbc_offshift)
    return bond_vec, bond_dist, model.bond_expansion(bond_dist)


def _warp_norm(X: torch.Tensor) -> torch.Tensor:
    """Per-channel squared Frobenius norm of a warp-layout tensor (N, 3, 3, units)."""
    return (X * X).sum((-3, -2))


# ---------------------------------------------------------------------------
# Layer level
# ---------------------------------------------------------------------------


def test_embedding_layer_parity(structure):
    from matgl.ops import graph_transform

    model_pyg, model_warp = _pyg_warp_pair()
    g, _ = _batch(structure)
    bond_vec, bond_dist, edge_attr = _layer_inputs(model_pyg, g)
    state_feat = model_pyg.state_embedding(torch.tensor([0, 1], device=DEVICE))
    z = g.node_type

    X_pyg, _ = model_pyg.tensor_embedding(z, g.edge_index, edge_attr, bond_dist, bond_vec, state_feat, g.batch)
    row_data, _, row_indptr, *_ = graph_transform(g.edge_index.int(), z.shape[0])
    X_warp = model_warp.tensor_embedding(
        z, g.edge_index, bond_dist, bond_vec, edge_attr, row_data, row_indptr, state_feat, g.batch
    )
    assert torch.allclose(_warp_norm(X_warp), tensor_norm(X_pyg), atol=1e-5, rtol=1e-4)

    # The state must actually reach the warp embedding.
    X_warp_other = model_warp.tensor_embedding(
        z, g.edge_index, bond_dist, bond_vec, edge_attr, row_data, row_indptr, state_feat.flip(0), g.batch
    )
    assert not torch.allclose(X_warp, X_warp_other)


def test_interaction_layer_parity(structure):
    from matgl.ops import graph_transform

    model_pyg, model_warp = _pyg_warp_pair()
    g, _ = _batch(structure)
    bond_vec, bond_dist, edge_attr = _layer_inputs(model_pyg, g)
    state_feat = model_pyg.state_embedding(torch.tensor([0, 1], device=DEVICE))
    z = g.node_type
    csr = graph_transform(g.edge_index.int(), z.shape[0])

    X_pyg, _ = model_pyg.tensor_embedding(z, g.edge_index, edge_attr, bond_dist, bond_vec, state_feat, g.batch)
    X_warp = model_warp.tensor_embedding(
        z, g.edge_index, bond_dist, bond_vec, edge_attr, csr[0], csr[2], state_feat, g.batch
    )
    Y_pyg = model_pyg.layers[0](g.edge_index, bond_dist, edge_attr, X_pyg, state_feat, g.batch)
    Y_warp = model_warp.layers[0](X_warp, g.edge_index, bond_dist, edge_attr, *csr, state_feat, g.batch)
    assert torch.allclose(_warp_norm(Y_warp), tensor_norm(Y_pyg), atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Model level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group", ["O(3)", "SO(3)"])
@pytest.mark.parametrize("is_intensive", [False, True])
def test_tensornet_parity_mixed_batch(structure, group, is_intensive):
    model_pyg, model_warp = _pyg_warp_pair(equivariance_invariance_group=group, is_intensive=is_intensive)
    g, _ = _batch(structure)
    state = torch.tensor([0, 1], device=DEVICE)
    out_pyg = model_pyg(g=g, state_attr=state)
    state_pyg = model_pyg.feature_dict["state_feat"]
    out_warp = model_warp(g=g, state_attr=state)
    assert not torch.allclose(out_pyg[0], out_pyg[1]), "fidelity label must change the output"
    assert torch.allclose(out_warp, out_pyg, atol=1e-5, rtol=1e-4)
    assert torch.allclose(model_warp.feature_dict["state_feat"], state_pyg, atol=1e-5, rtol=1e-4)


def test_warp_mixed_batch_matches_single(structure):
    """Per-graph state routing on the warp path: a mixed-fidelity batch equals the graphs run one at a time."""
    _, model_warp = _pyg_warp_pair()
    g, _ = _batch(structure)
    out = model_warp(g=g, state_attr=torch.tensor([0, 1], device=DEVICE))
    for i, label in enumerate((0, 1)):
        single, _ = _batch(structure, n=1)
        ref = model_warp(g=single, state_attr=torch.tensor([label], device=DEVICE))
        assert torch.allclose(out[i], ref, atol=1e-5, rtol=1e-4)


def test_qet_potential_parity(structure):
    """Energies, forces and stresses agree between backends for a mixed-fidelity batch."""
    model_pyg, model_warp = _pyg_warp_pair(QET)
    pot_pyg, pot_warp = Potential(model=model_pyg), Potential(model=model_warp)
    g, lattices = _batch(structure)
    state = torch.tensor([0, 1], device=DEVICE)
    e_pyg, f_pyg, s_pyg, *_ = pot_pyg(g.clone(), lattices, state)
    e_warp, f_warp, s_warp, *_ = pot_warp(g.clone(), lattices, state)
    assert not torch.allclose(e_pyg[0], e_pyg[1])
    assert torch.allclose(e_warp, e_pyg, atol=1e-5, rtol=1e-4)
    assert torch.allclose(f_warp, f_pyg, atol=1e-4)
    assert torch.allclose(s_warp, s_pyg, atol=1e-3)
    assert torch.allclose(model_warp.feature_dict["charge"], model_pyg.feature_dict["charge"], atol=1e-5)


@pytest.mark.parametrize("model_cls", [TensorNet, QET])
def test_training_gradients_parity(structure, model_cls):
    """Weight gradients of an energy + force loss (double backward) agree, including the state parameters."""
    model_pyg, model_warp = _pyg_warp_pair(model_cls)
    state = torch.tensor([0, 1], device=DEVICE)

    def grads(model):
        g, _ = _batch(structure)
        g.pos.requires_grad_(True)
        g.pbc_offshift = g.pbc_offshift.detach()
        energy = model(g=g, state_attr=state).sum()
        forces = -torch.autograd.grad(energy, g.pos, create_graph=True)[0]
        loss = energy + forces.pow(2).sum()
        names, params = zip(*[(n, p) for n, p in model.named_parameters() if p.requires_grad], strict=True)
        return dict(zip(names, torch.autograd.grad(loss, params, allow_unused=True), strict=True))

    grads_pyg, grads_warp = grads(model_pyg), grads(model_warp)
    state_names = [n for n in grads_pyg if "state" in n]
    assert state_names
    for name in state_names:
        assert grads_warp[name] is not None, f"{name} gets no gradient on the warp path"
        assert grads_warp[name].abs().sum() > 0, f"{name} gets a zero gradient on the warp path"
        assert torch.allclose(grads_warp[name], grads_pyg[name], atol=1e-4, rtol=1e-3), name
    # distance_proj is fused on the warp side; compare everything else that has a counterpart.
    for name, grad in grads_pyg.items():
        if name in grads_warp and grad is not None:
            assert torch.allclose(grads_warp[name], grad, atol=1e-4, rtol=1e-3), name


def test_warp_save_load_roundtrip(structure, tmp_path):
    _, model_warp = _pyg_warp_pair(QET)
    g, _ = _batch(structure, n=1)
    state = torch.tensor([1], device=DEVICE)
    ref = model_warp(g=g, state_attr=state)
    model_warp.cpu().save(tmp_path)
    loaded = QET.load(tmp_path).to(DEVICE)
    assert loaded._use_warp
    assert torch.allclose(loaded(g=g, state_attr=state), ref, atol=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason="Known gap: the PyG TensorEmbedding cannot load the warp layer's fused distance_proj weights, so a "
    "warp-trained checkpoint does not load on a machine without warp (or for LAMMPS export).",
)
def test_warp_checkpoint_loads_into_pyg():
    _, model_warp = _pyg_warp_pair()
    model_pyg = TensorNet(element_types=ELEMENTS, use_warp=False, **MFI_KWARGS)
    model_pyg.load_state_dict(model_warp.state_dict())
