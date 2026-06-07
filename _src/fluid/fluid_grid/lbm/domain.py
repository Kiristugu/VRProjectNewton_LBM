# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""LBM domain with double-buffered state (DESIGN.md §5.4)."""

from __future__ import annotations

from wanphys._src.core.domain import Domain

import warp as wp

from .model import FluidGridLbmModel
from .solver import FluidGridLbmSolver
from .state import FluidGridLbmState


class FluidGridLbmDomain(Domain):
    """D3Q19-BGK fluid grid domain.

    Encapsulates model, solver, and ping-pong ``state_in`` / ``state_out`` buffers.
    Intra-step ``f`` / ``F`` swap is handled inside ``FluidGridLbmSolver.step()``.
    """

    def __init__(
        self,
        model: FluidGridLbmModel,
        solver: FluidGridLbmSolver | None = None,
    ) -> None:
        self._model: FluidGridLbmModel = model
        self._solver: FluidGridLbmSolver = solver or FluidGridLbmSolver(model)
        self._state_in: FluidGridLbmState | None = None
        self._state_out: FluidGridLbmState | None = None

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
        assert self._state_in is not None
        return self._state_in

    def create_state(self) -> FluidGridLbmState:
        """Allocate internal double buffers; call ``init_uniform`` to set ICs."""
        self._state_in = FluidGridLbmState(self._model)
        self._state_out = FluidGridLbmState(self._model)
        return self._state_in

    def init_uniform(self, rho: float, u: wp.vec3) -> None:
        """Initialize both ping-pong buffers to a uniform equilibrium state."""
        if self._state_in is None or self._state_out is None:
            self.create_state()
        assert self._state_in is not None and self._state_out is not None
        self._solver.init_uniform(self._state_in, rho, u)
        self._solver.init_uniform(self._state_out, rho, u)

    def step(
        self,
        dt: float,
        contacts=None,
    ) -> None:
        if self._state_in is None or self._state_out is None:
            self.create_state()
        assert self._state_in is not None and self._state_out is not None
        self._solver.step(self._state_in, self._state_out, dt)
        self._state_in, self._state_out = self._state_out, self._state_in
