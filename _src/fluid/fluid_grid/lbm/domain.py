# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""LBM simulation domain with double-buffered state (DESIGN.md §5.4)."""

from __future__ import annotations

import warp as wp

from ..base import FluidGridDomainBase
from .model import FluidGridLbmModel
from .solver import FluidGridLbmSolver
from .state import FluidGridLbmState


class FluidGridLbmDomain(FluidGridDomainBase):
    """D3Q19-BGK domain; owns model, solver, and ping-pong LBM states."""

    def __init__(
        self,
        model: FluidGridLbmModel,
        solver: FluidGridLbmSolver | None = None,
    ) -> None:
        super().__init__(model, solver or FluidGridLbmSolver(model))

    @property
    def name(self) -> str:
        return "fluid_grid_lbm"

    @property
    def model(self) -> FluidGridLbmModel:
        return self._model

    @property
    def solver(self) -> FluidGridLbmSolver:
        return self._solver

    @property
    def state(self) -> FluidGridLbmState:
        if self._state_in is None:
            self.create_state()
        return self._state_in

    def create_state(self) -> FluidGridLbmState:
        self._state_in = FluidGridLbmState(self._model)
        self._state_out = FluidGridLbmState(self._model)
        return self._state_in

    def bake_box(self, center: wp.vec3, half_extents: wp.vec3) -> None:
        """Bake a box obstacle on both ping-pong states."""
        if self._state_in is None:
            self.create_state()
        assert self._state_in is not None and self._state_out is not None
        self._solver.bake_box(self._state_in, center, half_extents)
        self._solver.bake_box(self._state_out, center, half_extents)

    def bake_cylinder_z(self, center_x: float, center_y: float, radius: float) -> None:
        """Bake a z-aligned cylinder on both ping-pong states."""
        if self._state_in is None:
            self.create_state()
        assert self._state_in is not None and self._state_out is not None
        self._solver.bake_cylinder_z(self._state_in, center_x, center_y, radius)
        self._solver.bake_cylinder_z(self._state_out, center_x, center_y, radius)

    def bake_cylinder_y(self, center_x: float, center_z: float, radius: float) -> None:
        """Bake a y-aligned (infinite-height) cylinder on both ping-pong states."""
        if self._state_in is None:
            self.create_state()
        assert self._state_in is not None and self._state_out is not None
        self._solver.bake_cylinder_y(self._state_in, center_x, center_z, radius)
        self._solver.bake_cylinder_y(self._state_out, center_x, center_z, radius)

    def bake_solid_j_min(self, j_min: int) -> None:
        """Mark j >= j_min solid on both states (air above free surface)."""
        if self._state_in is None:
            self.create_state()
        assert self._state_in is not None and self._state_out is not None
        self._solver.bake_solid_j_min(self._state_in, j_min)
        self._solver.bake_solid_j_min(self._state_out, j_min)
