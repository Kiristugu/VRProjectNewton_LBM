# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""LBM simulation domain with double-buffered state (DESIGN.md §5.4)."""

from __future__ import annotations

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
