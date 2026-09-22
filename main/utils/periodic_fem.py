"""Mesh and periodic-boundary utilities for 2D image-based FEM analysis."""

from typing import Literal

import numpy as np
from scipy.sparse import coo_matrix


LoadMode = Literal["tension_x", "tension_y", "shear"]


def create_node_grid(image: np.ndarray, cell_length: float = 1.0) -> np.ndarray:
    """Create SolidSpy node coordinates and zero-valued boundary flags."""
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("image must be a square 2D array")
    if cell_length <= 0:
        raise ValueError("cell_length must be greater than zero")

    nodes_per_side = image.shape[0] + 1
    coordinates = np.linspace(0, cell_length, nodes_per_side)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    node_ids = np.arange(nodes_per_side**2)
    boundary_flags = np.zeros((node_ids.size, 2))
    return np.column_stack((node_ids, x_grid.ravel(), y_grid.ravel(), boundary_flags))


def create_element_topology(image: np.ndarray) -> np.ndarray:
    """Create SolidSpy quadrilateral elements and assign image material IDs."""
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("image must be a square 2D array")

    elements_per_side = image.shape[0]
    nodes_per_side = elements_per_side + 1
    row, column = np.indices((elements_per_side, elements_per_side))
    lower_left = row * nodes_per_side + column
    connectivity = np.column_stack(
        (
            lower_left.ravel(),
            (lower_left + 1).ravel(),
            (lower_left + nodes_per_side + 1).ravel(),
            (lower_left + nodes_per_side).ravel(),
        )
    )
    element_ids = np.arange(connectivity.shape[0])
    element_types = np.ones(connectivity.shape[0])
    material_ids = np.flip(image, axis=0).ravel()
    return np.column_stack((element_ids, element_types, material_ids, connectivity))


def reassign_periodic_dofs(
    total_dofs: int,
    image_dofs: list[int],
    reference_dofs: list[int],
) -> list[int]:
    """Return reduced DOF numbers after periodic image DOFs are removed."""
    if len(image_dofs) != len(reference_dofs):
        raise ValueError("image_dofs and reference_dofs must have the same length")

    reference_by_image = dict(zip(image_dofs, reference_dofs))
    reduced_dofs = []
    removed_dofs = 0
    for dof in range(total_dofs):
        if dof in reference_by_image:
            removed_dofs += 1
            reference_dof = reference_by_image[dof]
            reduced_dofs.append(
                reference_dof if reference_dof < dof else reference_dof - removed_dofs
            )
        else:
            reduced_dofs.append(dof - removed_dofs)
    return reduced_dofs


def identify_periodic_nodes(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Identify matching left/right, bottom/top, and corner node pairs."""
    x_min, x_max = np.min(nodes[:, 1]), np.max(nodes[:, 1])
    y_min, y_max = np.min(nodes[:, 2]), np.max(nodes[:, 2])
    left = nodes[nodes[:, 1] == x_min][:, 0]
    right = nodes[nodes[:, 1] == x_max][:, 0]
    bottom = nodes[nodes[:, 2] == y_min][:, 0]
    top = nodes[nodes[:, 2] == y_max][:, 0]

    # Pair complete left/right faces and pair only interior bottom/top nodes.
    # This covers every periodic constraint once and avoids duplicated corners.
    reference_nodes = np.concatenate((left, bottom[1:-1])).astype(int)
    image_nodes = np.concatenate((right, top[1:-1])).astype(int)
    return reference_nodes, image_nodes


def create_node_dof_map(node_ids: np.ndarray) -> tuple[list[int], dict[int, list[int]]]:
    """Map each node ID to its two displacement DOFs."""
    dof_map = {int(node): [2 * int(node), 2 * int(node) + 1] for node in node_ids}
    dofs = [dof for node_dofs in dof_map.values() for dof in node_dofs]
    return dofs, dof_map


def build_periodic_transformation(
    base_displacement: float,
    total_dofs: int,
    reference_nodes: np.ndarray,
    image_nodes: np.ndarray,
    reference_dofs: list[int],
    image_dofs: list[int],
    reference_dof_map: dict[int, list[int]],
    image_dof_map: dict[int, list[int]],
    reduced_dofs: list[int],
    nodes: np.ndarray | None = None,
    mode: LoadMode = "tension_x",
    displacement: float = 0.64,
):
    """Build the sparse periodic transformation matrix and prescribed vector."""
    if mode not in {"tension_x", "tension_y", "shear"}:
        raise ValueError(f"Unsupported load mode: {mode}")

    rows, columns, values = [], [], []
    prescribed = np.zeros(total_dofs)
    image_dof_set = set(image_dofs)

    for dof in range(total_dofs):
        if dof not in image_dof_set:
            rows.append(dof)
            columns.append(reduced_dofs[dof])
            values.append(1)

    for image_node, reference_node in zip(image_nodes, reference_nodes):
        for reference_dof, image_dof in zip(
            reference_dof_map[int(reference_node)], image_dof_map[int(image_node)]
        ):
            rows.append(image_dof)
            columns.append(reduced_dofs[reference_dof])
            values.append(1)
            prescribed[image_dof] = base_displacement

    if nodes is not None:
        x_max, x_min = np.max(nodes[:, 1]), np.min(nodes[:, 1])
        y_max, y_min = np.max(nodes[:, 2]), np.min(nodes[:, 2])
        for node in image_nodes:
            ux, uy = image_dof_map[int(node)]
            x, y = nodes[int(node), 1], nodes[int(node), 2]
            if mode == "tension_x" and np.isclose(x, x_max):
                prescribed[ux] += displacement
            elif mode == "tension_y" and np.isclose(y, y_max):
                prescribed[uy] += displacement
            elif mode == "shear":
                if np.isclose(y, y_max):
                    prescribed[ux] = displacement
                elif np.isclose(y, y_min):
                    prescribed[ux] = 0.0

    matrix_shape = (total_dofs, total_dofs - len(reference_dofs))
    transformation = coo_matrix(
        (values, (rows, columns)), shape=matrix_shape
    ).tocsr()
    return transformation, prescribed


def solve_periodic_load_case(
    stiffness_matrix,
    boundary_conditions: np.ndarray,
    nodes: np.ndarray,
    elements: np.ndarray,
    materials: np.ndarray,
    transformation,
    prescribed: np.ndarray,
):
    """Solve one periodic load case and return displacement, strain, and stress."""
    import solidspy.postprocesor as postprocesor
    import solidspy.solutil as solutil

    reduced_stiffness = transformation.T @ stiffness_matrix @ transformation
    reduced_rhs = transformation.T @ (-stiffness_matrix @ prescribed)
    reduced_displacement = solutil.static_sol(
        reduced_stiffness.tocsr(), reduced_rhs
    )
    displacement = transformation @ reduced_displacement + prescribed
    complete_displacement = postprocesor.complete_disp(
        boundary_conditions, nodes, displacement
    )
    strain_nodes, stress_nodes = postprocesor.strain_nodes(
        nodes, elements, materials, complete_displacement
    )
    return complete_displacement, strain_nodes, stress_nodes
