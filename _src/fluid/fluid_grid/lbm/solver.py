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
        self._refresh_bc_velocities()

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
        """Bake an axis-aligned box obstacle into ``state.solid`` (lattice-unit center/extents)."""
        wp.launch(
            kernels.bake_box,
            dim=self._grid_dim,
            inputs=[state.solid, state.solid_phi, center, half_extents],
            device=self.device,
        )

    def bake_cylinder_z(
        self,
        state: FluidGridLbmState,
        center_x: float,
        center_y: float,
        radius: float,
    ) -> None:
        """Bake a z-aligned cylinder (2D cylinder extruded through all k layers)."""
        wp.launch(
            kernels.bake_cylinder_z,
            dim=self._grid_dim,
            inputs=[state.solid, state.solid_phi, center_x, center_y, radius],
            device=self.device,
        )

    def bake_cylinder_y(
        self,
        state: FluidGridLbmState,
        center_x: float,
        center_z: float,
        radius: float,
    ) -> None:
        """Bake a y-aligned cylinder (infinite-height pillar, circular cross-section in x-z)."""
        wp.launch(
            kernels.bake_cylinder_y,
            dim=self._grid_dim,
            inputs=[state.solid, state.solid_phi, center_x, center_z, radius],
            device=self.device,
        )

    def bake_solid_j_min(self, state: FluidGridLbmState, j_min: int) -> None:
        """Mark j >= j_min as solid (air above nominal free surface)."""
        wp.launch(
            kernels.bake_solid_j_min,
            dim=self._grid_dim,
            inputs=[state.solid, state.solid_phi, j_min],
            device=self.device,
        )

    def configure_channel_flow(self, u_in: float) -> None:
        """Horizontal channel: inlet velocity (x=0), pressure outlet (x=nx-1), no-slip top/bottom, periodic z."""
        u_tuple = (float(u_in), 0.0, 0.0)
        zero = (0.0, 0.0, 0.0)
        self.model.bc_x_left = 2
        self.model.bc_x_right = 1
        self.model.bc_y_left = 2
        self.model.bc_y_right = 2
        self.model.bc_z_left = 0
        self.model.bc_z_right = 0
        self.model.bc_vel_x_left = u_tuple
        self.model.bc_vel_x_right = zero
        self.model.bc_vel_y_left = zero
        self.model.bc_vel_y_right = zero
        self.model.bc_vel_z_left = zero
        self.model.bc_vel_z_right = zero
        self.model.bc_velocity = u_tuple
        self.model.bc_rho = 1.0
        self._refresh_bc_velocities()

    def init_uniform_flow(self, state: FluidGridLbmState, rho: float, u: wp.vec3) -> None:
        """Initialize uniform density and velocity at equilibrium (channel inflow IC)."""
        self.init_uniform(state, rho, u)

    def bake_sphere(self, state: FluidGridLbmState, center: wp.vec3, radius: float) -> None:
        """Placeholder for spherical obstacle baking."""
        del state, center, radius
