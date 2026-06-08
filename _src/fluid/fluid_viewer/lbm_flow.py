# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""OpenGL visualization helpers for LBM grid velocity fields."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from wanphys._src.fluid.fluid_grid.lbm.domain import FluidGridLbmDomain


def _arr_pointer(arr: np.ndarray):
    return arr.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class LbmScalarVolumeRenderer:
    """Ray-marched 3D scalar volume (|u|) composited over the GL scene."""

    _fullscreen_vertex_shader = """
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aUV;
    out vec2 vUV;
    void main()
    {
        vUV = aUV;
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
    """

    _composite_fragment_shader = """
    #version 330 core
    in vec2 vUV;

    uniform sampler2D scene_color_tex;
    uniform sampler3D scalar_tex;

    uniform vec3 camera_pos_world;
    uniform vec3 camera_front_world;
    uniform vec3 camera_right_world;
    uniform vec3 camera_up_world;
    uniform vec3 volume_min_world;
    uniform vec3 volume_max_world;
    uniform vec3 color_low;
    uniform vec3 color_high;
    uniform float scalar_cutoff;
    uniform float max_scalar;
    uniform float absorption;
    uniform float opacity_boost;
    uniform float tan_half_fov;
    uniform float aspect_ratio;
    uniform float step_size_world;
    uniform int max_steps;

    out vec4 FragColor;

    bool intersect_aabb(vec3 ray_origin, vec3 ray_dir, vec3 box_min, vec3 box_max, out float t_min, out float t_exit)
    {
        vec3 safe_dir = vec3(
            abs(ray_dir.x) < 1.0e-6 ? (ray_dir.x >= 0.0 ? 1.0e-6 : -1.0e-6) : ray_dir.x,
            abs(ray_dir.y) < 1.0e-6 ? (ray_dir.y >= 0.0 ? 1.0e-6 : -1.0e-6) : ray_dir.y,
            abs(ray_dir.z) < 1.0e-6 ? (ray_dir.z >= 0.0 ? 1.0e-6 : -1.0e-6) : ray_dir.z
        );
        vec3 inv_dir = 1.0 / safe_dir;
        vec3 t0 = (box_min - ray_origin) * inv_dir;
        vec3 t1 = (box_max - ray_origin) * inv_dir;
        vec3 tsmaller = min(t0, t1);
        vec3 tbigger = max(t0, t1);
        t_min = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
        t_exit = min(min(tbigger.x, tbigger.y), tbigger.z);
        return t_exit >= max(t_min, 0.0);
    }

    vec3 world_to_uvw(vec3 world_pos)
    {
        return (world_pos - volume_min_world) / max(volume_max_world - volume_min_world, vec3(1.0e-6));
    }

    float sample_scalar(vec3 world_pos)
    {
        vec3 uvw = world_to_uvw(world_pos);
        if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0))))
            return 0.0;
        return texture(scalar_tex, uvw).r;
    }

    vec3 velocity_colormap(float t)
    {
        t = clamp(t, 0.0, 1.0);
        if (t < 0.5)
            return mix(color_low, vec3(0.15, 0.85, 0.95), t * 2.0);
        return mix(vec3(0.15, 0.85, 0.95), color_high, (t - 0.5) * 2.0);
    }

    void main()
    {
        vec3 scene_color = texture(scene_color_tex, vUV).rgb;
        vec3 ray_origin = camera_pos_world;
        vec2 ndc = vUV * 2.0 - 1.0;
        vec3 ray_dir = normalize(
            camera_front_world
            + camera_right_world * ndc.x * tan_half_fov * aspect_ratio
            + camera_up_world * ndc.y * tan_half_fov
        );

        float t_enter;
        float t_exit;
        if (!intersect_aabb(ray_origin, ray_dir, volume_min_world, volume_max_world, t_enter, t_exit))
        {
            FragColor = vec4(scene_color, 1.0);
            return;
        }

        float t = max(t_enter, 0.0);
        float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
        t += jitter * step_size_world;

        float transmittance = 1.0;
        vec3 accumulated = vec3(0.0);

        for (int i = 0; i < 192; ++i)
        {
            if (i >= max_steps || t >= t_exit || transmittance <= 0.02)
                break;

            vec3 world_pos = ray_origin + ray_dir * t;
            float scalar = sample_scalar(world_pos);
            if (scalar > scalar_cutoff)
            {
                float norm = scalar / max(max_scalar, 1.0e-6);
                float extinction = scalar * absorption;
                float sample_alpha = 1.0 - exp(-extinction * step_size_world * opacity_boost);
                vec3 sample_color = velocity_colormap(norm);
                accumulated += transmittance * sample_alpha * sample_color;
                transmittance *= 1.0 - sample_alpha;
            }
            t += step_size_world;
        }

        FragColor = vec4(scene_color * transmittance + accumulated, 1.0);
    }
    """

    def __init__(
        self,
        viewer,
        world_min: tuple[float, float, float],
        world_max: tuple[float, float, float],
        grid_resolution: tuple[int, int, int],
    ) -> None:
        self.viewer = viewer
        self.world_min = np.asarray(world_min, dtype=np.float32)
        self.world_max = np.asarray(world_max, dtype=np.float32)
        self.grid_resolution = tuple(int(v) for v in grid_resolution)

        self.scalar_cutoff = 0.002
        self.max_scalar = 0.12
        self.absorption = 1.4
        self.opacity_boost = 1.6
        self.color_low = np.array((0.05, 0.08, 0.55), dtype=np.float32)
        self.color_high = np.array((0.95, 0.15, 0.05), dtype=np.float32)

        self._gl = None
        self._initialized = False
        self._failed = False
        self._program = None
        self._scalar_tex = None
        self._uniforms: dict[str, int] = {}
        self._host_scalar: np.ndarray | None = None

    @property
    def available(self) -> bool:
        return not self._failed

    def set_speed_field(self, speed: np.ndarray) -> None:
        """Upload |u| field with shape (nx, ny, nz)."""
        self._host_scalar = np.ascontiguousarray(speed, dtype=np.float32)
        peak = float(np.max(self._host_scalar)) if self._host_scalar.size else 0.0
        if peak > 1.0e-6:
            self.max_scalar = max(peak, 0.05)

    def render(self, viewer) -> None:
        del viewer
        if self._failed or self._host_scalar is None:
            return
        try:
            self._ensure_initialized()
            self._upload_texture()
            self._composite()
        except Exception as exc:
            self._failed = True
            print(f"[LbmVolume] disabling velocity volume rendering: {exc}")

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        from pyglet import gl
        from pyglet.graphics.shader import Shader, ShaderProgram

        self._gl = gl
        self._program = ShaderProgram(
            Shader(self._fullscreen_vertex_shader, "vertex"),
            Shader(self._composite_fragment_shader, "fragment"),
        )
        self._scalar_tex = gl.GLuint()
        gl.glGenTextures(1, self._scalar_tex)
        self._allocate_texture()
        self._uniforms = {
            name: gl.glGetUniformLocation(self._program.id, ctypes.c_char_p(name.encode("utf-8")))
            for name in (
                "scene_color_tex",
                "scalar_tex",
                "camera_pos_world",
                "camera_front_world",
                "camera_right_world",
                "camera_up_world",
                "volume_min_world",
                "volume_max_world",
                "color_low",
                "color_high",
                "scalar_cutoff",
                "max_scalar",
                "absorption",
                "opacity_boost",
                "tan_half_fov",
                "aspect_ratio",
                "step_size_world",
                "max_steps",
            )
        }
        self._initialized = True

    def _allocate_texture(self) -> None:
        gl = self._gl
        nx, ny, nz = self.grid_resolution
        gl.glBindTexture(gl.GL_TEXTURE_3D, self._scalar_tex)
        gl.glTexImage3D(
            gl.GL_TEXTURE_3D,
            0,
            gl.GL_R32F,
            nx,
            ny,
            nz,
            0,
            gl.GL_RED,
            gl.GL_FLOAT,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_3D, 0)

    def _upload_texture(self) -> None:
        assert self._host_scalar is not None
        gl = self._gl
        nx, ny, nz = self.grid_resolution
        volume_host = np.transpose(self._host_scalar, (2, 1, 0))
        volume_host = np.ascontiguousarray(volume_host, dtype=np.float32)
        gl.glBindTexture(gl.GL_TEXTURE_3D, self._scalar_tex)
        gl.glTexSubImage3D(
            gl.GL_TEXTURE_3D,
            0,
            0,
            0,
            0,
            nx,
            ny,
            nz,
            gl.GL_RED,
            gl.GL_FLOAT,
            volume_host.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        gl.glBindTexture(gl.GL_TEXTURE_3D, 0)

    def _composite(self) -> None:
        gl = self._gl
        renderer = self.viewer.renderer
        camera = self.viewer.camera

        span = self.world_max - self.world_min
        step_size_world = float(np.min(span / np.maximum(np.asarray(self.grid_resolution, dtype=np.float32), 1.0)) * 0.9)
        max_steps = int(np.ceil(np.linalg.norm(span) / max(step_size_world, 1.0e-4)))
        max_steps = max(32, min(max_steps, 160))

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, int(renderer._screen_width), int(renderer._screen_height))
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glDisable(gl.GL_BLEND)
        gl.glUseProgram(self._program.id)

        uniforms = self._uniforms
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, renderer._frame_texture)
        gl.glUniform1i(uniforms["scene_color_tex"], 0)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_3D, self._scalar_tex)
        gl.glUniform1i(uniforms["scalar_tex"], 1)

        gl.glUniform3f(uniforms["camera_pos_world"], float(camera.pos[0]), float(camera.pos[1]), float(camera.pos[2]))
        camera_front = np.asarray(camera.get_front(), dtype=np.float32)
        camera_right = np.asarray(camera.get_right(), dtype=np.float32)
        camera_up = np.asarray(camera.get_up(), dtype=np.float32)
        gl.glUniform3f(uniforms["camera_front_world"], float(camera_front[0]), float(camera_front[1]), float(camera_front[2]))
        gl.glUniform3f(uniforms["camera_right_world"], float(camera_right[0]), float(camera_right[1]), float(camera_right[2]))
        gl.glUniform3f(uniforms["camera_up_world"], float(camera_up[0]), float(camera_up[1]), float(camera_up[2]))
        gl.glUniform3f(uniforms["volume_min_world"], *self.world_min)
        gl.glUniform3f(uniforms["volume_max_world"], *self.world_max)
        gl.glUniform3f(uniforms["color_low"], *self.color_low)
        gl.glUniform3f(uniforms["color_high"], *self.color_high)
        gl.glUniform1f(uniforms["scalar_cutoff"], self.scalar_cutoff)
        gl.glUniform1f(uniforms["max_scalar"], self.max_scalar)
        gl.glUniform1f(uniforms["absorption"], self.absorption)
        gl.glUniform1f(uniforms["opacity_boost"], self.opacity_boost)
        gl.glUniform1f(uniforms["tan_half_fov"], float(np.tan(np.deg2rad(float(camera.fov)) * 0.5)))
        gl.glUniform1f(uniforms["aspect_ratio"], float(renderer._screen_width) / max(float(renderer._screen_height), 1.0))
        gl.glUniform1f(uniforms["step_size_world"], step_size_world)
        gl.glUniform1i(uniforms["max_steps"], max_steps)

        gl.glBindVertexArray(renderer._frame_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, len(renderer._frame_indices), gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_3D, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glUseProgram(0)
        gl.glDepthMask(gl.GL_TRUE)


class LbmCavityVisualizer:
    """Composite LBM cavity view: domain wireframe + |u| volume (+ optional mid-plane vectors)."""

    def __init__(
        self,
        viewer,
        domain: FluidGridLbmDomain,
        *,
        cell_size: float = 1.0,
        show_volume: bool = True,
        show_boundary: bool = True,
        show_mid_plane_vectors: bool = True,
        vector_stride: int = 4,
        vector_scale: float = 0.8,
    ) -> None:
        self.viewer = viewer
        self.domain = domain
        self.cell_size = float(cell_size)
        self.show_volume = show_volume
        self.show_boundary = show_boundary
        self.show_mid_plane_vectors = show_mid_plane_vectors
        self.vector_stride = max(1, int(vector_stride))
        self.vector_scale = float(vector_scale)

        model = domain.model
        self.nx: int = model.nx
        self.ny: int = model.ny
        self.nz: int = model.nz
        self.device = model._device

        cs = self.cell_size
        self.world_min = (0.0, 0.0, 0.0)
        self.world_max = (self.nx * cs, self.ny * cs, self.nz * cs)

        self._volume: LbmScalarVolumeRenderer | None = None
        if self.show_volume:
            self._volume = LbmScalarVolumeRenderer(
                viewer,
                self.world_min,
                self.world_max,
                (self.nx, self.ny, self.nz),
            )

        if self.show_boundary:
            self._init_boundary_lines()

        if hasattr(viewer, "register_post_render_callback") and self._volume is not None:
            viewer.register_post_render_callback(lambda current_viewer: self._volume.render(current_viewer))

    def _init_boundary_lines(self) -> None:
        cs = self.cell_size
        mn = np.array(self.world_min, dtype=np.float32)
        mx = np.array(self.world_max, dtype=np.float32)
        edges: list[tuple[np.ndarray, np.ndarray]] = [
            (mn, np.array([mx[0], mn[1], mn[2]], dtype=np.float32)),
            (np.array([mx[0], mn[1], mn[2]], dtype=np.float32), np.array([mx[0], mn[1], mx[2]], dtype=np.float32)),
            (np.array([mx[0], mn[1], mx[2]], dtype=np.float32), np.array([mn[0], mn[1], mx[2]], dtype=np.float32)),
            (np.array([mn[0], mn[1], mx[2]], dtype=np.float32), mn),
            (np.array([mn[0], mx[1], mn[2]], dtype=np.float32), mx),
            (mx, np.array([mn[0], mx[1], mx[2]], dtype=np.float32)),
            (np.array([mn[0], mx[1], mx[2]], dtype=np.float32), np.array([mn[0], mx[1], mn[2]], dtype=np.float32)),
            (mn, np.array([mn[0], mx[1], mn[2]], dtype=np.float32)),
            (np.array([mx[0], mn[1], mn[2]], dtype=np.float32), np.array([mx[0], mx[1], mn[2]], dtype=np.float32)),
            (np.array([mx[0], mn[1], mx[2]], dtype=np.float32), np.array([mx[0], mx[1], mx[2]], dtype=np.float32)),
            (np.array([mn[0], mn[1], mx[2]], dtype=np.float32), np.array([mn[0], mx[1], mx[2]], dtype=np.float32)),
        ]
        starts = wp.array([e[0] for e in edges], dtype=wp.vec3, device=self.device)
        ends = wp.array([e[1] for e in edges], dtype=wp.vec3, device=self.device)
        colors = wp.full(len(edges), wp.vec3(0.55, 0.55, 0.6), dtype=wp.vec3, device=self.device)
        self._boundary_starts = starts
        self._boundary_ends = ends
        self._boundary_colors = colors

    def update(self) -> None:
        v_np: np.ndarray = self.domain.state.v.numpy()
        speed: np.ndarray = np.linalg.norm(v_np, axis=-1).astype(np.float32)
        if self._volume is not None and self._volume.available:
            self._volume.set_speed_field(speed)

    def render(self) -> None:
        self.update()
        if self.show_boundary:
            self.viewer.log_lines(
                "lbm/boundary",
                starts=self._boundary_starts,
                ends=self._boundary_ends,
                colors=self._boundary_colors,
                width=0.004,
                hidden=False,
            )
        if self.show_mid_plane_vectors:
            self._render_mid_plane_vectors()

    def _render_mid_plane_vectors(self) -> None:
        v_np: np.ndarray = self.domain.state.v.numpy()
        cs = self.cell_size
        mid_j: int = self.ny // 2
        stride = self.vector_stride

        starts_list: list[np.ndarray] = []
        ends_list: list[np.ndarray] = []
        colors_list: list[np.ndarray] = []

        for i in range(0, self.nx, stride):
            for k in range(0, self.nz, stride):
                vel = v_np[i, mid_j, k]
                speed = float(np.linalg.norm(vel))
                if speed < 1.0e-5:
                    continue
                origin = np.array([(i + 0.5) * cs, (mid_j + 0.5) * cs, (k + 0.5) * cs], dtype=np.float32)
                tip = origin + vel.astype(np.float32) * self.vector_scale
                starts_list.append(origin)
                ends_list.append(tip)
                ref_speed = self._volume.max_scalar if self._volume is not None else 0.1
                t = min(1.0, speed / max(ref_speed, 1.0e-6))
                colors_list.append(np.array([t, 0.25 * (1.0 - t), 1.0 - t], dtype=np.float32))

        if not starts_list:
            self.viewer.log_lines("lbm/mid_plane_vectors", None, None, None)
            return

        starts = wp.array(starts_list, dtype=wp.vec3, device=self.device)
        ends = wp.array(ends_list, dtype=wp.vec3, device=self.device)
        colors = wp.array(colors_list, dtype=wp.vec3, device=self.device)
        self.viewer.log_lines(
            "lbm/mid_plane_vectors",
            starts=starts,
            ends=ends,
            colors=colors,
            width=0.003,
            hidden=False,
        )

    def setup_camera(self) -> None:
        cs = self.cell_size
        cx = 0.5 * self.nx * cs
        cy = 0.5 * self.ny * cs
        cz = 0.5 * self.nz * cs
        dist = max(self.nx, self.ny, self.nz) * cs * 1.8
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(
                pos=wp.vec3(cx + dist * 0.65, cy + dist * 0.45, cz + dist * 0.75),
                pitch=-22.0,
                yaw=-125.0,
            )
