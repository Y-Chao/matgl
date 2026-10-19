"""Tests for the QET model."""

from __future__ import annotations

import os

import pytest
import torch
from pymatgen.core import Lattice, Structure

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph._compute import compute_pair_vector_and_distance
from matgl.models._qet import QET


def _make_qet(**overrides):
    """Construct QET, suppressing the warp kernel so the pure-PyTorch path runs."""
    overrides.setdefault("use_warp", False)
    return QET(**overrides)


def _single_atom_graph(cutoff=5.0):
    """Build a PyG graph for a single-atom structure, mirroring ``conftest.get_graph``."""
    structure = Structure(Lattice.cubic(3.17), ["Mo"], [[0.0, 0.0, 0.0]])
    element_types = get_element_list([structure])
    converter = Structure2Graph(element_types=element_types, cutoff=cutoff)
    graph, lattice, state = converter.get_graph(structure)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lattice[0])
    graph.pos = graph.frac_coords @ lattice[0]
    bond_vec, bond_dist = compute_pair_vector_and_distance(graph.pos, graph.edge_index, graph.pbc_offshift)
    graph.bond_vec = bond_vec
    graph.bond_dist = bond_dist
    return element_types, graph, state


def test_qet(graph_MoS):
    """Forward across activations + save/load + SO(3) variant."""
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    _, graph, _ = graph_MoS

    activations = ["swish", "tanh", "sigmoid", "softplus2", "softexp"]

    for act in activations:
        model = _make_qet(is_intensive=False, activation_type=act)
        output = model(g=graph, total_charge=torch.tensor([0.0]))
        assert torch.numel(output) == 1

    model.save(".")
    QET.load(".")
    for fname in ("model.pt", "model.json", "state.pt"):
        os.remove(fname)

    model = _make_qet(is_intensive=False, equivariance_invariance_group="SO(3)")
    output = model(g=graph, total_charge=torch.tensor([0.0]))
    assert torch.numel(output) == 1


def test_qet_return_features(graph_MoS):
    """`return_features=True` returns (node_feat, atomic_energies) with the right shapes."""
    torch.manual_seed(0)
    _, graph, _ = graph_MoS
    model = _make_qet(is_intensive=False, return_features=True)
    node_feat, atomic_energies = model(g=graph, total_charge=torch.tensor([0.0]))
    n_nodes = graph.pos.shape[0]
    # +1 charge, +1 elec_pot
    assert node_feat.shape == (n_nodes, model.units + 2)
    assert atomic_energies.shape[0] == n_nodes


def test_qet_include_magmom(graph_MoS):
    torch.manual_seed(0)
    _, graph, _ = graph_MoS
    model = _make_qet(is_intensive=False, include_magmom=True, return_features=True)
    node_feat, _ = model(g=graph, total_charge=torch.tensor([0.0]))
    n_nodes = graph.pos.shape[0]
    # +1 charge, +1 elec_pot, +1 magmom
    assert node_feat.shape == (n_nodes, model.units + 3)


def test_qet_is_hardness_envs(graph_MoS):
    torch.manual_seed(0)
    _, graph, _ = graph_MoS
    model = _make_qet(is_intensive=False, is_hardness_envs=True)
    output = model(g=graph, total_charge=torch.tensor([0.0]))
    assert torch.numel(output) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"include_magmom": True},
        {"is_hardness_envs": True},
        {"is_sigma_train": True},
        {"equivariance_invariance_group": "SO(3)"},
    ],
)
def test_qet_single_atom(overrides):
    """Single-atom systems must not collapse the node dim via ``torch.squeeze``.

    Regression test: with one atom the per-node ``sigma`` tensor has shape
    ``(1,)``; a bare ``torch.squeeze`` collapsed it to a 0-d scalar, which then
    raised ``IndexError`` inside ``ElectrostaticPotential.forward``.
    """
    torch.manual_seed(0)
    element_types, graph, _ = _single_atom_graph()
    model = _make_qet(element_types=element_types, is_intensive=False, **overrides)
    output = model(g=graph, total_charge=torch.tensor([0.0]))
    assert torch.numel(output) == 1
    assert torch.isfinite(output).all()


# --- Multi-fidelity (global state) support -----------------------------------------------------------------------
# Ko & Ong, npj Comput. Mater. 11, 65 (2025): an integer fidelity label is embedded as a learned global state that
# conditions every interaction block and is updated from the node features after each block.

_MFI_KWARGS = {"include_state": True, "ntypes_state": 2, "dim_state_embedding": 16}


def _mfi_inputs(structure, element_types=("Mo", "S")):
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)
    graph, lattice, _ = converter.get_graph(structure)
    return graph, lattice


def _distorted(structure):
    """Break the symmetry of a fixture so forces are non-zero."""
    s = structure.copy()
    s.translate_sites([0], [0.12, -0.07, 0.05], frac_coords=False)
    return s


def _mfi_potential(**overrides):
    from matgl.apps.pes import Potential

    torch.manual_seed(0)
    model = _make_qet(element_types=("Mo", "S"), **{**_MFI_KWARGS, **overrides})
    return Potential(model=model)


def test_qet_mfi_state_changes_energy_and_forces(MoS):
    pot = _mfi_potential()
    graph, lattice = _mfi_inputs(_distorted(MoS))
    e0, f0, *_ = pot(graph, lattice, torch.tensor([0]))
    e1, f1, *_ = pot(graph, lattice, torch.tensor([1]))
    assert not torch.allclose(e0, e1), "fidelity label must change the predicted energy"
    assert not torch.allclose(f0, f1), "fidelity label must change the predicted forces"
    # Fidelity must be registered as trainable state parameters, not a silent no-op.
    state_params = [n for n, _ in pot.model.named_parameters() if "state" in n]
    assert state_params


def test_qet_mfi_mixed_batch_matches_single(MoS):
    from torch_geometric.data import Batch

    pot = _mfi_potential()
    graph, lattice = _mfi_inputs(_distorted(MoS))
    singles = [pot(graph, lattice, torch.tensor([s])) for s in (0, 1)]
    batch = Batch.from_data_list([graph, graph])
    e, f, *_ = pot(batch, torch.cat([lattice, lattice]), torch.tensor([0, 1]))
    assert torch.allclose(e, torch.cat([singles[0][0].reshape(1), singles[1][0].reshape(1)]), atol=1e-5)
    assert torch.allclose(f, torch.cat([singles[0][1], singles[1][1]]), atol=1e-5)


def test_qet_mfi_rotation_invariance(MoS):
    import numpy as np

    pot = _mfi_potential()
    MoS = _distorted(MoS)
    graph, lattice = _mfi_inputs(MoS)
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    rotated = Structure(Lattice(MoS.lattice.matrix @ rot.T), MoS.species, MoS.frac_coords)
    graph_r, lattice_r = _mfi_inputs(rotated)
    for s in (0, 1):
        e, f, *_ = pot(graph, lattice, torch.tensor([s]))
        e_r, f_r, *_ = pot(graph_r, lattice_r, torch.tensor([s]))
        assert torch.allclose(e, e_r, atol=1e-5)
        assert torch.allclose(f @ torch.tensor(rot.T, dtype=f.dtype), f_r, atol=1e-4)


def test_qet_mfi_forces_match_finite_difference(MoS):
    pot = _mfi_potential()
    pot = pot.double()
    graph, lattice = _mfi_inputs(_distorted(MoS))
    graph.frac_coords = graph.frac_coords.double()
    lattice = lattice.double()
    state = torch.tensor([1])
    _, f, *_ = pot(graph, lattice, state)
    h = 1e-4
    inv_lat = torch.linalg.inv(lattice[0])
    for atom, axis in [(0, 0), (1, 2)]:
        energies = []
        for sign in (1, -1):
            disp = torch.zeros(3, dtype=torch.float64)
            disp[axis] = sign * h
            g = graph.clone()
            g.frac_coords = graph.frac_coords.clone()
            g.frac_coords[atom] += disp @ inv_lat
            energies.append(pot(g, lattice, state)[0])
        fd_force = -(energies[0] - energies[1]) / (2 * h)
        assert torch.allclose(f[atom, axis], fd_force, atol=1e-4)


def test_qet_mfi_save_load_roundtrip(MoS, tmp_path):
    torch.manual_seed(0)
    model = _make_qet(element_types=("Mo", "S"), **_MFI_KWARGS)
    graph, lattice = _mfi_inputs(MoS)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lattice[0])
    graph.pos = graph.frac_coords @ lattice[0]
    ref = model(g=graph, state_attr=torch.tensor([1]))
    model.save(tmp_path)
    loaded = QET.load(tmp_path)
    assert torch.allclose(ref, loaded(g=graph, state_attr=torch.tensor([1])))


def test_qet_mfi_requires_one_label_per_graph(graph_MoS):
    _, graph, _ = graph_MoS
    model = _make_qet(**_MFI_KWARGS)
    with pytest.raises(ValueError, match="state_attr"):
        model(g=graph)
    # The converter default ``[0.0, 0.0]`` is two values for one graph, which must not be silently accepted.
    with pytest.raises(ValueError, match="state_attr"):
        model(g=graph, state_attr=torch.tensor([0.0, 0.0]))


def test_qet_mfi_requires_ntypes_state():
    with pytest.raises(ValueError, match="ntypes_state"):
        _make_qet(include_state=True, dim_state_embedding=16)


def test_qet_without_state_has_no_state_params(graph_MoS):
    _, graph, _ = graph_MoS
    model = _make_qet()
    assert not [n for n, _ in model.named_parameters() if "state" in n]
    # ``state_attr`` is ignored when ``include_state=False`` (e.g. the converter default ``[0.0, 0.0]``).
    torch.manual_seed(0)
    out_a = model(g=graph, total_charge=torch.tensor([0.0]))
    out_b = model(g=graph, total_charge=torch.tensor([0.0]), state_attr=torch.tensor([0.0, 0.0]))
    assert torch.allclose(out_a, out_b)


@pytest.mark.parametrize("nblocks", [1, 3])
def test_qet_mfi_all_parameters_receive_gradients(MoS, nblocks):
    """No dead state parameters (required for DDP with find_unused_parameters=False)."""
    pot = _mfi_potential(nblocks=nblocks)
    graph, lattice = _mfi_inputs(_distorted(MoS))
    e, f, *_ = pot(graph, lattice, torch.tensor([1]))
    (e.sum() + f.pow(2).sum()).backward()
    no_grad = [n for n, p in pot.model.named_parameters() if p.requires_grad and p.grad is None]
    state_no_grad = [n for n in no_grad if "state" in n]
    assert not state_no_grad, state_no_grad
