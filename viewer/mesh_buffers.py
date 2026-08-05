import ctypes
import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COLOR_ARRAY,
    GL_DYNAMIC_DRAW,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FLOAT,
    GL_NORMAL_ARRAY,
    GL_STATIC_DRAW,
    GL_TEXTURE_COORD_ARRAY,
    GL_TRIANGLES,
    GL_UNSIGNED_INT,
    GL_VERTEX_ARRAY,
    glBindBuffer,
    glBufferData,
    glBufferSubData,
    glColorPointer,
    glColor3f,
    glDeleteBuffers,
    glDisableClientState,
    glDrawElements,
    glEnableClientState,
    glGenBuffers,
    glNormalPointer,
    glTexCoordPointer,
    glVertexPointer,
)


class MeshBuffers:
    def __init__(self):
        self.position_buf = 0
        self.normal_buf = 0
        self.color_buf = 0
        self.texcoord_buf = 0
        self.index_buf = 0
        self.index_count = 0
        self.vertex_count = 0
        self.has_texcoords = False

    def is_valid(self):
        return self.index_buf != 0 and self.index_count > 0

    def build(self, positions, normals, colors, indices, texcoords=None):
        self.release()

        positions = np.ascontiguousarray(positions, dtype=np.float32)
        normals = np.ascontiguousarray(normals, dtype=np.float32)
        colors = np.ascontiguousarray(colors, dtype=np.float32)
        indices = np.ascontiguousarray(indices, dtype=np.uint32)

        self.vertex_count = len(positions)
        self.index_count = len(indices)

        self.position_buf = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.position_buf)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_DYNAMIC_DRAW)

        self.normal_buf = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.normal_buf)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_DYNAMIC_DRAW)

        self.color_buf = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.color_buf)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)

        if texcoords is not None and len(texcoords) == self.vertex_count:
            texcoords = np.ascontiguousarray(texcoords, dtype=np.float32)
            self.texcoord_buf = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.texcoord_buf)
            glBufferData(GL_ARRAY_BUFFER, texcoords.nbytes, texcoords, GL_STATIC_DRAW)
            self.has_texcoords = True
        else:
            self.has_texcoords = False

        self.index_buf = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_buf)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def update(self, positions, normals):
        if not self.is_valid():
            return

        positions = np.ascontiguousarray(positions, dtype=np.float32)
        normals = np.ascontiguousarray(normals, dtype=np.float32)

        if len(positions) != self.vertex_count:
            return

        glBindBuffer(GL_ARRAY_BUFFER, self.position_buf)
        glBufferSubData(GL_ARRAY_BUFFER, 0, positions.nbytes, positions)

        glBindBuffer(GL_ARRAY_BUFFER, self.normal_buf)
        glBufferSubData(GL_ARRAY_BUFFER, 0, normals.nbytes, normals)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _enable_arrays(self, with_texcoords, with_colors=True):
        use_uv = with_texcoords and self.has_texcoords

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        if with_colors:
            glEnableClientState(GL_COLOR_ARRAY)
        else:
            glDisableClientState(GL_COLOR_ARRAY)
            glColor3f(1.0, 1.0, 1.0)
        if use_uv:
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, self.position_buf)
        glVertexPointer(3, GL_FLOAT, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, self.normal_buf)
        glNormalPointer(GL_FLOAT, 0, None)

        if with_colors:
            glBindBuffer(GL_ARRAY_BUFFER, self.color_buf)
            glColorPointer(3, GL_FLOAT, 0, None)

        if use_uv:
            glBindBuffer(GL_ARRAY_BUFFER, self.texcoord_buf)
            glTexCoordPointer(2, GL_FLOAT, 0, None)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_buf)
        return use_uv

    def _disable_arrays(self, use_uv):
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        if use_uv:
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)

    def draw(self, with_texcoords=False, with_colors=True):
        if not self.is_valid():
            return
        use_uv = self._enable_arrays(with_texcoords, with_colors)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        self._disable_arrays(use_uv)

    def draw_range(self, offset, count, with_texcoords=False, with_colors=True):
        if not self.is_valid() or count <= 0:
            return
        use_uv = self._enable_arrays(with_texcoords, with_colors)
        glDrawElements(
            GL_TRIANGLES, count, GL_UNSIGNED_INT,
            ctypes.c_void_p(offset * 4)
        )
        self._disable_arrays(use_uv)

    def release(self):
        for buffer_id in (
            self.position_buf,
            self.normal_buf,
            self.color_buf,
            self.texcoord_buf,
            self.index_buf,
        ):
            if buffer_id:
                glDeleteBuffers(1, [buffer_id])

        self.position_buf = 0
        self.normal_buf = 0
        self.color_buf = 0
        self.texcoord_buf = 0
        self.index_buf = 0
        self.index_count = 0
        self.vertex_count = 0
        self.has_texcoords = False