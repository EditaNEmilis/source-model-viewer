import numpy as np

from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COLOR_ARRAY,
    GL_DYNAMIC_DRAW,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FLOAT,
    GL_NORMAL_ARRAY,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    GL_UNSIGNED_INT,
    GL_VERTEX_ARRAY,
    glBindBuffer,
    glBufferData,
    glBufferSubData,
    glColorPointer,
    glDeleteBuffers,
    glDisableClientState,
    glDrawElements,
    glEnableClientState,
    glGenBuffers,
    glNormalPointer,
    glVertexPointer,
)


class MeshBuffers:
    def __init__(self):
        self.position_buf = 0
        self.normal_buf = 0
        self.color_buf = 0
        self.index_buf = 0
        self.index_count = 0
        self.vertex_count = 0

    def is_valid(self):
        return self.index_buf != 0 and self.index_count > 0

    def build(self, positions, normals, colors, indices):
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

    def draw(self):
        if not self.is_valid():
            return

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, self.position_buf)
        glVertexPointer(3, GL_FLOAT, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, self.normal_buf)
        glNormalPointer(GL_FLOAT, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, self.color_buf)
        glColorPointer(3, GL_FLOAT, 0, None)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_buf)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    def release(self):
        for buffer_id in (
            self.position_buf,
            self.normal_buf,
            self.color_buf,
            self.index_buf,
        ):
            if buffer_id:
                glDeleteBuffers(1, [buffer_id])

        self.position_buf = 0
        self.normal_buf = 0
        self.color_buf = 0
        self.index_buf = 0
        self.index_count = 0
        self.vertex_count = 0