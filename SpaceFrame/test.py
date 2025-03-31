
import glfw
from OpenGL.GL import *
import numpy as np


vertexShaderSource = "#version 460 core\n layout (location = 0) in vec3 aPos;\n void main()\n{\n gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);}\0"
fragmentShaderSource = "#version 460 core\n out vec4 FragColor;\n void main()\n {\n FragColor = vec4(0.8f, 0.3f, 0.02f, 1.0f);\n}\n\0"

if not glfw.init():
    raise Error("Failed to initialize GLFW")

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

vertices = np.array([
    -0.5, -0.5*3**0.5/3, 0.0,
    0.5, -0.5*3**0.5/3, 0.0,
    0.0, 0.5*3**0.5*2/3, 0.0,
    -0.5/2, 0.5*3**0.5/6, 0.0,
    0.5/2, 0.5*3**0.5/6, 0.0,
    0.0, -0.5*3**0.5/3, 0.0
    ], GLfloat)

indices = np.array([
    0, 3, 5,
    3, 2, 4, 
    5, 4, 1
], GLuint)


window = glfw.create_window(800, 800, "Window", None, None)

if window == None:
    glfw.terminate()
    raise Error("Failed to create GLFW window")

glfw.make_context_current(window)

glViewport(0, 0, 800, 800) 

vertexShader = glCreateShader(GL_VERTEX_SHADER)
glShaderSource(vertexShader, vertexShaderSource)
glCompileShader(vertexShader)

fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
glShaderSource(fragmentShader, fragmentShaderSource)
glCompileShader(fragmentShader)

shaderProgram = glCreateProgram()
glAttachShader(shaderProgram, vertexShader)
glAttachShader(shaderProgram, fragmentShader)
glLinkProgram(shaderProgram)

glDeleteShader(vertexShader)
glDeleteShader(fragmentShader)

VAO = glGenVertexArrays(1)
VBO = glGenBuffers(1)
EBO = glGenBuffers(1)

glBindVertexArray(VAO)

glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), None
glVertexAttribPointer()
glEnableVertexAttribArray(0)

glBindBuffer(GL_ARRAY_BUFFER, 0)
glBindVertexArray(0)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

glClearColor(0.0, 0.2 , 0.3, 1.0)
glClear(GL_COLOR_BUFFER_BIT)
glfw.swap_buffers(window)

while not glfw.window_should_close(window):
    glClearColor(0.0, 0.2 , 0.3, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(shaderProgram)
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, 9, GL_UNSIGNED_INT, None)
    glfw.swap_buffers(window)
    glfw.poll_events()

glDeleteVertexArrays(1, [VAO])
glDeleteBuffers(1, [VBO])
glDeleteBuffers(1, [EBO])
glDeleteProgram(shaderProgram)
glfw.destroy_window(window)
glfw.terminate()
