# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""LBM time integrator — orchestrates collide / stream / BC / macro (DESIGN.md §4)."""

from __future__ import annotations

from typing import Any

import warp as wp

from ..base import FluidGridSolverBase
from . import kernels
from .model import FluidGridLbmModel
from .state import FluidGridLbmState


class FluidGridLbmSolver(FluidGridSolverBase):
    """D3Q19-BGK solver; week-1 minimal pipeline for M1 rest-fluid validation."""

    def __init__(self, model: FluidGridLbmModel) -> None:
        self.model = model
        self.nx = model.nx
        self.ny = model.ny
        self.nz = model.nz
        self.device = model._device
        self._grid_dim = (self.nx, self.ny, self.nz)
        self._force = wp.vec3(model.force[0], model.force[1], model.force[2])
        self._use_guo = 1 if model.use_guo_force else 0

    def step(
        self,
        state_in: FluidGridLbmState,
        state_out: FluidGridLbmState,
        dt: float,
        contacts: Any | None = None,
        control: Any | None = None,
    ) -> None:
        """Advance one LBM step: collide → stream → BC → macro (DESIGN.md §4)."""
        del dt, contacts, control

        wp.launch(
            kernels.collide_bgk,
            dim=self._grid_dim,
            inputs=[
                state_in.f,
                state_out.F,
                state_in.rho,
                state_in.v,
                state_in.solid,
                self.model.omega,
            ],
            device=self.device,
        )

        # Week 1: identity stream placeholder (member B replaces with stream_pull).
        wp.launch(
            kernels.stream_pull_identity,
            dim=self._grid_dim,
            inputs=[state_out.F, state_in.solid],
            device=self.device,
        )

        wp.launch(
            kernels.apply_boundaries,
            dim=self._grid_dim,
            inputs=[
                state_out.F,
                state_in.f,
                state_in.rho,
                state_in.v,
                state_in.solid,
            ],
            device=self.device,
        )

        wp.launch(
            kernels.update_macro,
            dim=self._grid_dim,
            inputs=[
                state_out.F,
                state_out.f,
                state_out.rho,
                state_out.v,
                state_in.solid,
                self._force,
                self._use_guo,
            ],
            device=self.device,
        )

        wp.copy(state_out.solid, state_in.solid)

    def init_uniform(self, state: FluidGridLbmState, rho: float, u: wp.vec3) -> None:
        """Initialize uniform density and velocity at equilibrium."""
        wp.launch(
            kernels.init_equilibrium,
            dim=self._grid_dim,
            inputs=[state.f, state.F, state.rho, state.v, state.solid, rho, u],
            device=self.device,
        )

    def set_lid_velocity(self, u_lid: wp.vec3) -> None:
        """Configure top-wall velocity BC (used by cavity example in week 2)."""
        self.model.bc_y_right = 2
        self.model.bc_velocity = (float(u_lid[0]), float(u_lid[1]), float(u_lid[2]))

    def bake_box(self, state: FluidGridLbmState, center: wp.vec3, half_extents: wp.vec3) -> None:
        """Placeholder for obstacle baking (member C, week 3)."""
        del state, center, half_extents

    def bake_sphere(self, state: FluidGridLbmState, center: wp.vec3, radius: float) -> None:
        """Placeholder for obstacle baking (member C, week 3)."""
        del state, center, radius
