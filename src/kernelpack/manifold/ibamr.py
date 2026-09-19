from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import (
    SphericalSBFModel,
    SurfaceGeometry,
    build_spherical_sbf_model,
    evaluate_spherical_sbf_field,
    evaluate_spherical_sbf_geometry,
)
from .transfer import farthest_point_subset


def _table(path: Path) -> np.ndarray:
    table = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(table)


def _columns(table: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([table[name] for name in names])


@dataclass
class IBAMRSurfaceTrajectory:
    """Lazy reader and reduced-control SBF model for the public RBC trajectory."""

    folder: str | Path
    xi: int = 6
    control_point_count: int | None = None

    def __post_init__(self) -> None:
        self.folder = Path(self.folder)
        required = ("material.csv", "faces.csv", "diagnostics.csv")
        missing = [name for name in required if not (self.folder / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"missing IBAMR trajectory files in {self.folder}: {', '.join(missing)}"
            )
        material = _table(self.folder / "material.csv")
        order = np.argsort(material["node_id"].astype(int))
        self.material_sites = _columns(material, ("u_x", "u_y", "u_z"))[order]
        self.material_sites /= np.linalg.norm(self.material_sites, axis=1, keepdims=True)
        diagnostics = _table(self.folder / "diagnostics.csv")
        self.steps = diagnostics["step"].astype(int)
        self.times = diagnostics["time"].astype(float)
        count = self.control_point_count
        if count is None:
            count = min(self.material_sites.shape[0], max(64, int(np.ceil(self.material_sites.shape[0] / 5))))
        self.control_indices = farthest_point_subset(self.material_sites, count=int(count))
        initial_positions, _ = self.native_frame(0)
        self.model: SphericalSBFModel = build_spherical_sbf_model(
            initial_positions[self.control_indices],
            self.material_sites,
            parameter_sites=self.material_sites[self.control_indices],
        )

    def native_frame(self, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        path = self.folder / f"frame_{int(self.steps[frame_index]):06d}.csv"
        frame = _table(path)
        order = np.argsort(frame["node_id"].astype(int))
        positions = _columns(frame, ("x", "y", "z"))[order]
        velocities = _columns(frame, ("u", "v", "w"))[order]
        return positions, velocities

    def geometry(self, frame_index: int) -> tuple[SurfaceGeometry, np.ndarray]:
        positions, velocities = self.native_frame(frame_index)
        geometry = evaluate_spherical_sbf_geometry(
            self.model, positions[self.control_indices]
        )
        velocity = evaluate_spherical_sbf_field(
            self.model,
            velocities[self.control_indices],
            self.material_sites,
        )
        return geometry, velocity
