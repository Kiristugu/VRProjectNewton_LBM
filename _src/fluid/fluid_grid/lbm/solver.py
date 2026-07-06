# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""LBM time integrator — orchestrates collide / stream / BC / macro (DESIGN.md §4)."""

from __future__ import annotations

from typing import Any

import warp as wp

import numpy as np

from ..base import FluidGridSolverBase
from . import kernels
from .model import FluidGridLbmModel
from .mrt import INV_M_MATRIX, M_MATRIX, build_relaxation_rates
from .state import FluidGridLbmState


class FluidGridLbmSolver(FluidGridSolverBase):
    """D3Q19 LBM solver (BGK / TRT / MRT) with collide → copy → stream_pull → BC → macro."""

    def __init__(self, model: FluidGridLbmModel) -> None:
        self.model = model
        self.nx = model.nx
        self.ny = model.ny
        self.nz = model.nz
        self.device = model._device
        self._grid_dim = (self.nx, self.ny, self.nz)
        self._force = wp.vec3(model.force[0], model.force[1], model.force[2])
        self._use_guo = 1 if model.use_guo_force else 0
        self._refresh_bc_velocities()
        self._mrt_m_matrix = wp.array(M_MATRIX.astype(np.float32), dtype=float, device=self.device)
        self._mrt_inv_m_matrix = wp.array(INV_M_MATRIX.astype(np.float32), dtype=float, device=self.device)
        self._mrt_s_diag = wp.array(
            build_relaxation_rates(model.omega, model.mrt_ghost_s).astype(np.float32),
            dtype=float,
            device=self.device,
        )

    def _vec3_from_tuple(self, components: tuple[float, float, float]) -> wp.vec3:
        return wp.vec3(components[0], components[1], components[2])

    def _refresh_bc_velocities(self) -> None:
        """Cache per-face BC velocities as wp.vec3 for kernel launch."""
        m = self.model
        self._bc_vel_x_left = self._vec3_from_tuple(m.bc_vel_x_left)
        self._bc_vel_x_right = self._vec3_from_tuple(m.bc_vel_x_right)
        self._bc_vel_y_left = self._vec3_from_tuple(m.bc_vel_y_left)
        self._bc_vel_y_right = self._vec3_from_tuple(m.bc_vel_y_right)
        self._bc_vel_z_left = self._vec3_from_tuple(m.bc_vel_z_left)
        self._bc_vel_z_right = self._vec3_from_tuple(m.bc_vel_z_right)

    def step(
        self,
        state_in: FluidGridLbmState,
        state_out: FluidGridLbmState,
        dt: float,
        contacts: Any | None = None,
        control: Any | None = None,
    ) -> None:
        """Advance one LBM step: collide → stream → BC → macro → swap (DESIGN.md §4)."""
        del dt, contacts, control

        if self.model.collide_impl == "trt":
            # TRT: tau_minus = 0.5 + Lambda * (tau_plus - 0.5)
            tau_plus = self.model.tau
            tau_minus = 0.5 + float(self.model.trt_lambda) * (tau_plus - 0.5)
            omega_plus = self.model.omega
            omega_minus = 1.0 / tau_minus
            wp.launch(
                kernels.collide_trt,
                dim=self._grid_dim,
                inputs=[
                    state_in.f,
                    state_out.F,
                    state_in.rho,
                    state_in.v,
                    state_in.solid,
                    omega_plus,
                    omega_minus,
                ],
                device=self.device,
            )
        elif self.model.collide_impl == "mrt":
            wp.launch(
                kernels.collide_mrt,
                dim=self._grid_dim,
                inputs=[
                    state_in.f,
                    state_out.F,
                    state_in.rho,
                    state_in.v,
                    state_in.solid,
                    self._mrt_m_matrix,
                    self._mrt_inv_m_matrix,
                    self._mrt_s_diag,
                ],
                device=self.device,
            )
        else:
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

        # Post-collision distributions live in F; copy to f for pull read (see docs/phase1_stream_pull.md).
        wp.copy(state_out.f, state_out.F)

        wp.launch(
            kernels.stream_pull,
            dim=self._grid_dim,
            inputs=[
                state_out.f,
                state_out.F,
                state_in.solid,
                self.nx,
                self.ny,
                self.nz,
                self.model.bc_x_left,
                self.model.bc_x_right,
                self.model.bc_y_left,
                self.model.bc_y_right,
                self.model.bc_z_left,
                self.model.bc_z_right,
            ],
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
                self.nx,
                self.ny,
                self.nz,
                self.model.bc_x_left,
                self.model.bc_x_right,
                self.model.bc_y_left,
                self.model.bc_y_right,
                self.model.bc_z_left,
                self.model.bc_z_right,
                self.model.bc_rho,
                self._bc_vel_x_left,
                self._bc_vel_x_right,
                self._bc_vel_y_left,
                self._bc_vel_y_right,
                self._bc_vel_z_left,
                self._bc_vel_z_right,
            ],
            device=self.device,
        )

        wp.launch(
            kernels.update_macro,
            dim=self._grid_dim,
            inputs=[
                state_out.F,
                state_out.rho,
                state_out.v,
                state_in.solid,
                self._force,
                self._use_guo,
            ],
            device=self.device,
        )

        self._swap_buffers(state_out)
        wp.copy(state_out.solid, state_in.solid)

    def _swap_buffers(self, state: FluidGridLbmState) -> None:
        """Exchange f and F after macro so f holds the post-step distributions (DESIGN.md §4 step 5)."""
        state.f, state.F = state.F, state.f

    def init_uniform(self, state: FluidGridLbmState, rho: float, u: wp.vec3) -> None:
        """Initialize uniform density and velocity at equilibrium."""
        wp.launch(
            kernels.init_equilibrium,
            dim=self._grid_dim,
            inputs=[state.f, state.F, state.rho, state.v, state.solid, rho, u],
            device=self.device,
        )

    def configure_cavity_walls(self) -> None:
        """No-slip box: all six faces velocity BC with u=0 (DESIGN.md §10.3, phase2_boundaries.md)."""
        zero = (0.0, 0.0, 0.0)
        self.model.bc_x_left = 2
        self.model.bc_x_right = 2
        self.model.bc_y_left = 2
        self.model.bc_y_right = 2
        self.model.bc_z_left = 2
        self.model.bc_z_right = 2
        self.model.bc_vel_x_left = zero
        self.model.bc_vel_x_right = zero
        self.model.bc_vel_y_left = zero
        self.model.bc_vel_y_right = zero
        self.model.bc_vel_z_left = zero
        self.model.bc_vel_z_right = zero
        self.model.bc_velocity = zero
        self._refresh_bc_velocities()

    def set_lid_velocity(self, u_lid: wp.vec3) -> None:
        """Set moving lid on y = ny-1 face (bc_y_right, bc_vel_y_right)."""
        u_tuple = (float(u_lid[0]), float(u_lid[1]), float(u_lid[2]))
        self.model.bc_y_right = 2
        self.model.bc_vel_y_right = u_tuple
        self.model.bc_velocity = u_tuple
        self._refresh_bc_velocities()

    def bake_box(self, state: FluidGridLbmState, center: wp.vec3, half_extents: wp.vec3) -> None:
        """Bake axis-aligned box obstacle into ``state.solid``."""
        wp.launch(
            kernels.bake_box,
            dim=self._grid_dim,
            inputs=[state.solid, center, half_extents],
            device=self.device,
        )

    def bake_sphere(self, state: FluidGridLbmState, center: wp.vec3, radius: float) -> None:
        """Placeholder for obstacle baking (member C, week 3)."""
        del state, center, radius