# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Runtime state for D3Q19-BGK lattice Boltzmann fluid grid (week-1 minimal)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..base import FluidGridStateBase
from .lattice import Q

if TYPE_CHECKING:
    from .model import FluidGridLbmModel


class FluidGridLbmState(FluidGridStateBase):
    """LBM distribution functions and macroscopic fields on a regular grid."""

    def __init__(self, model: FluidGridLbmModel, requires_grad: bool = False) -> None:
        super().__init__(model, requires_grad)

        nx, ny, nz = self.res
        self.f = wp.zeros((nx, ny, nz, Q), dtype=float, device=self.device, requires_grad=requires_grad)
        self.F = wp.zeros((nx, ny, nz, Q), dtype=float, device=self.device, requires_grad=requires_grad)
        self.rho = wp.zeros((nx, ny, nz), dtype=float, device=self.device, requires_grad=requires_grad)
        self.v = wp.zeros((nx, ny, nz), dtype=wp.vec3, device=self.device, requires_grad=requires_grad)
        self.solid = wp.zeros((nx, ny, nz), dtype=wp.int32, device=self.device)

    def clear_forces(self) -> None:
        """CompositeSimulation contract; LBM body force lives on the model."""

    def clear(self) -> None:
        super().clear()
        self.f.zero_()
        self.F.zero_()
        self.rho.zero_()
        self.v.zero_()
        self.solid.zero_()

    def clone(self) -> FluidGridLbmState:
        new_state = FluidGridLbmState(self.model, requires_grad=self.requires_grad)
        wp.copy(new_state.f, self.f)
        wp.copy(new_state.F, self.F)
        wp.copy(new_state.rho, self.rho)
        wp.copy(new_state.v, self.v)
        wp.copy(new_state.solid, self.solid)
        wp.copy(new_state.vel_u, self.vel_u)
        wp.copy(new_state.vel_v, self.vel_v)
        wp.copy(new_state.vel_w, self.vel_w)
        wp.copy(new_state.pressure, self.pressure)
        wp.copy(new_state.density, self.density)
        wp.copy(new_state.solid_phi, self.solid_phi)
        wp.copy(new_state.solid_body_id, self.solid_body_id)
        return new_state
