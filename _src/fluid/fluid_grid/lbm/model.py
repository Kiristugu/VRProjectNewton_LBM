# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Static configuration for D3Q19-BGK lattice Boltzmann fluid grid."""

from __future__ import annotations

from dataclasses import dataclass

from ..base import FluidGridModelBase


@dataclass
class FluidGridLbmModel(FluidGridModelBase):
    """LBM model parameters in lattice units (dx = dt = 1)."""

    nu: float = 0.16667
    rho0: float = 1.0
    force: tuple[float, float, float] = (0.0, 0.0, 0.0)
    use_guo_force: bool = True

    # Face BC types: 0=periodic, 1=pressure (Zou-He rho), 2=velocity (Zou-He u)
    bc_x_left: int = 0
    bc_x_right: int = 0
    bc_y_left: int = 0
    bc_y_right: int = 0
    bc_z_left: int = 0
    bc_z_right: int = 0

    bc_rho: float = 1.0
    bc_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def tau(self) -> float:
        # nu = c_s^2 (tau - 0.5), c_s^2 = 1/3 in D3Q19 lattice units
        return 3.0 * self.nu + 0.5

    @property
    def omega(self) -> float:
        return 1.0 / self.tau
