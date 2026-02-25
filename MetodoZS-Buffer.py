# MetodoS_Buffer_final.py
# Z-Buffer (GPU) vs S-Buffer (CPU) 
# Requisitos: PyQt5, PyOpenGL, numpy
# JOSE ARIZA

import sys
import ctypes
import numpy as np
from PyQt5 import QtWidgets, QtOpenGL, QtCore
from OpenGL.GL import *

# -----------------------
# SHADERS (Phong & Gouraud)
# -----------------------
VERTEX_GOURAUD = """#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec3 color;
uniform mat4 model; uniform mat4 view; uniform mat4 projection;
uniform vec3 lightPos; uniform vec3 viewPos; uniform vec3 lightColor; uniform float shininess;
out vec3 v_color;
void main(){
    mat3 nmat = transpose(inverse(mat3(model)));
    vec3 N = normalize(nmat * normal);
    vec3 fragPos = vec3(model * vec4(position, 1.0));
    vec3 ambient = 0.1 * lightColor * color;
    vec3 L = normalize(lightPos - fragPos);
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * lightColor * color;
    vec3 V = normalize(viewPos - fragPos);
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(R, V), 0.0), shininess);
    vec3 specular = spec * lightColor;
    v_color = ambient + diffuse + specular;
    gl_Position = projection * view * model * vec4(position, 1.0);
}"""

FRAGMENT_GOURAUD = """#version 330 core
in vec3 v_color;
out vec4 FragColor;
void main(){ FragColor = vec4(v_color,1.0); }"""

VERTEX_PHONG = """#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec3 color;
uniform mat4 model; uniform mat4 view; uniform mat4 projection;
out vec3 fragPos; out vec3 N; out vec3 vertColor;
void main(){
    fragPos = vec3(model * vec4(position,1.0));
    N = mat3(transpose(inverse(model))) * normal;
    vertColor = color;
    gl_Position = projection * view * model * vec4(position,1.0);
}"""

FRAGMENT_PHONG = """#version 330 core
in vec3 fragPos; in vec3 N; in vec3 vertColor;
uniform vec3 lightPos; uniform vec3 viewPos; uniform vec3 lightColor; uniform float shininess;
out vec4 FragColor;
void main(){
    vec3 norm = normalize(N);
    vec3 L = normalize(lightPos - fragPos);
    vec3 V = normalize(viewPos - fragPos);
    vec3 ambient = 0.1 * lightColor * vertColor;
    float diff = max(dot(norm,L),0.0);
    vec3 diffuse = diff * lightColor * vertColor;
    vec3 R = reflect(-L,norm);
    float spec = pow(max(dot(R,V),0.0),shininess);
    vec3 specular = spec * lightColor;
    FragColor = vec4(ambient + diffuse + specular,1.0);
}"""

# -----------------------
# SHADER UTILS
# -----------------------
def compile_shader(src, stype):
    shader = glCreateShader(stype)
    glShaderSource(shader, src)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader

def link_program(vs, fs):
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(program).decode())
    return program

# -----------------------
# MATRICES (numpy-based, stable)
# -----------------------
def mat_identity():
    return np.eye(4, dtype=np.float32)

def mat_translation(x, y, z):
    M = np.eye(4, dtype=np.float32)
    M[0,3] = x
    M[1,3] = y
    M[2,3] = z
    return M

def mat_rotation_x(rad):
    c = np.cos(rad); s = np.sin(rad)
    M = np.eye(4, dtype=np.float32)
    M[1,1] = c; M[1,2] = -s
    M[2,1] = s; M[2,2] = c
    return M

def mat_rotation_y(rad):
    c = np.cos(rad); s = np.sin(rad)
    M = np.eye(4, dtype=np.float32)
    M[0,0] = c; M[0,2] = s
    M[2,0] = -s; M[2,2] = c
    return M

def mat_perspective(fov_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    M = np.zeros((4,4), dtype=np.float32)
    M[0,0] = f / aspect
    M[1,1] = f
    M[2,2] = (far + near) / (near - far)
    M[2,3] = (2 * far * near) / (near - far)
    M[3,2] = -1.0
    return M

def mat_lookat(eye, target, up):
    epsilon = 1e-12
    F = target - eye
    F = F / (np.linalg.norm(F) + epsilon)
    U = up / (np.linalg.norm(up) + epsilon)
    S = np.cross(F, U)
    S = S / (np.linalg.norm(S) + epsilon)
    U = np.cross(S, F)
    M = np.eye(4, dtype=np.float32)
    M[0,0:3] = S
    M[1,0:3] = U
    M[2,0:3] = -F
    T = mat_translation(-eye[0], -eye[1], -eye[2])
    return M @ T

# -----------------------
# GEOMETRÍA
# -----------------------
def create_cube():
    positions = [
        (-1,-1,1),(1,-1,1),(1,1,1),  (-1,-1,1),(1,1,1),(-1,1,1),
        (-1,-1,-1),(1,-1,-1),(1,1,-1),  (-1,-1,-1),(1,1,-1),(-1,1,-1),
        (-1,-1,-1),(-1,-1,1),(-1,1,1),  (-1,-1,-1),(-1,1,1),(-1,1,-1),
        (1,-1,-1),(1,1,-1),(1,1,1),  (1,-1,-1),(1,1,1),(1,-1,1),
        (-1,1,-1),(-1,1,1),(1,1,1),  (-1,1,-1),(1,1,1),(1,1,-1),
        (-1,-1,-1),(1,-1,-1),(1,-1,1), (-1,-1,-1),(1,-1,1),(-1,-1,1)
    ]
    normals=[]
    for i in range(0,36,3):
        p0=np.array(positions[i]); p1=np.array(positions[i+1]); p2=np.array(positions[i+2])
        n=np.cross(p1-p0,p2-p0)
        n=n/ (np.linalg.norm(n) + 1e-12)
        normals += [tuple(n)]*3
    colors=[(0.8,0.3,0.3)]*36
    data=[]
    for i in range(36):
        data += list(positions[i]) + list(normals[i]) + list(colors[i])
    return np.array(data, dtype=np.float32)

def create_plane():
    positions=[(-3,-1,-3),(3,-1,-3),(3,-1,3),(-3,-1,-3),(3,-1,3),(-3,-1,3)]
    normals=[(0,1,0)]*6
    colors=[(0.6,0.6,0.6)]*6
    data=[]
    for i in range(6):
        data += list(positions[i]) + list(normals[i]) + list(colors[i])
    return np.array(data, dtype=np.float32)

# -----------------------
# UTIL (Funciones esenciales para CPU Rasterizer)
# -----------------------
def clamp(x,a,b): return max(a,min(b,x))

def barycentric(p,a,b,c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    # Determinante (doble del área del triángulo 2D)
    den = v0[0]*v1[1] - v1[0]*v0[1]
    if abs(den) < 1e-9: return -1,-1,-1
    # Coordenadas baricéntricas (u, v, w) -> (w_0, w_1, w_2)
    v = (v2[0]*v1[1] - v1[0]*v2[1]) / den
    w = (v0[0]*v2[1] - v2[0]*v0[1]) / den
    u = 1.0 - v - w
    return u, v, w

# CORRECCIÓN: Faltaban operadores de multiplicación en la conversión NDC a pantalla
def ndc_to_screen(ndc_xy, width, height):
    # NDC [-1, 1] -> [0, 1] -> [0, width-1] (x)
    x = (ndc_xy[0] * 0.5 + 0.5) * (width - 1)
    # NDC [-1, 1] -> [0, 1] -> [0, height-1] (y)
    # Se añade (1.0 - ...) para invertir el eje Y (OpenGL vs. pantalla/NumPy)
    y = (1.0 - (ndc_xy[1] * 0.5 + 0.5)) * (height - 1)
    return np.array([x, y], dtype=np.float32)

# -----------------------
# OPENGL WIDGET
# -----------------------
class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rot_x = -20.0
        self.rot_y = 35.0
        self.distance = 8.0

        self.use_phong = True
        self.use_gpu_depth = True # Controla glEnable(GL_DEPTH_TEST)

        self.use_sbuffer = False # Controla si se usa el renderizado CPU
        self.sbuffer_scale = 1.0 # Permite reducir la resolución para mejorar rendimiento

        self.light_pos = np.array([2.0,4.0,2.0], dtype=np.float32)
        self.light_color = np.array([1.0,1.0,1.0], dtype=np.float32)
        self.shininess = 32.0

        self.last_pos = None

        self.cube = None
        self.plane = None

        self.prog_gouraud = None
        self.prog_phong = None

        # Buffers para el renderizado CPU
        self.cpu_fb = None # Frame buffer (color)
        self.sbuffer = None # S-Buffer (depth)
        self.sbuffer_changed_lines = set()
        self.tex_id = None # Textura para dibujar el resultado del CPU FB

        self.sbuffer_needs_update = True
        self.width = 0
        self.height = 0

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.15,0.15,0.17,1.0)

        vs_g = compile_shader(VERTEX_GOURAUD, GL_VERTEX_SHADER)
        fs_g = compile_shader(FRAGMENT_GOURAUD, GL_FRAGMENT_SHADER)
        self.prog_gouraud = link_program(vs_g, fs_g)

        vs_p = compile_shader(VERTEX_PHONG, GL_VERTEX_SHADER)
        fs_p = compile_shader(FRAGMENT_PHONG, GL_FRAGMENT_SHADER)
        self.prog_phong = link_program(vs_p, fs_p)

        self.plane = create_plane()
        self.cube = create_cube()

        stride = 9 * 4

        # Configuración de VBOs/VAOs para GPU
        self.cube_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.cube.nbytes, self.cube, GL_STATIC_DRAW)
        self.cube_vao = glGenVertexArrays(1)
        glBindVertexArray(self.cube_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_vbo)
        # Posición (0)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,False,stride,ctypes.c_void_p(0))
        # Normal (1)
        glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,False,stride,ctypes.c_void_p(12))
        # Color (2)
        glEnableVertexAttribArray(2); glVertexAttribPointer(2,3,GL_FLOAT,False,stride,ctypes.c_void_p(24))

        # Plane VAO/VBO
        self.plane_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.plane_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.plane.nbytes, self.plane, GL_STATIC_DRAW)
        self.plane_vao = glGenVertexArrays(1)
        glBindVertexArray(self.plane_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.plane_vbo)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,False,stride,ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,False,stride,ctypes.c_void_p(12))
        glEnableVertexAttribArray(2); glVertexAttribPointer(2,3,GL_FLOAT,False,stride,ctypes.c_void_p(24))

        # Textura para el Frame Buffer del CPU
        self.tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D, 0)

    def resizeGL(self, w, h):
        if h == 0: h = 1
        glViewport(0,0,w,h)
        self.width = w; self.height = h

        # Dimensiones para el S-Buffer (puede ser la mitad de la ventana)
        sw = max(1, int(self.width * self.sbuffer_scale))
        sh = max(1, int(self.height * self.sbuffer_scale))

        # Inicialización de buffers de CPU
        self.cpu_fb = np.zeros((sh, sw, 3), dtype=np.uint8)
        # El S-Buffer es una lista de diccionarios, donde cada dict representa una scanline
        # Almacena {x_coord: depth_value}
        self.sbuffer = [dict() for _ in range(sh)] 
        self.sbuffer_changed_lines = set()

        # Re-dimensionar la textura de OpenGL
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, sw, sh, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.sbuffer_needs_update = True

    def paintGL(self):
        # ----------------------------------------------------
        # 1. Pipeline S-Buffer (CPU)
        # ----------------------------------------------------
        if self.use_sbuffer:
            glDisable(GL_DEPTH_TEST)
            glClear(GL_COLOR_BUFFER_BIT)

            if self.sbuffer_needs_update:
                try:
                    self.render_sbuffer_to_cpu()
                except Exception as e:
                    print("Error en render_sbuffer_to_cpu:", e)
                    self.cpu_fb.fill(0)
                self.sbuffer_needs_update = False

            # Dibuja el resultado del Frame Buffer (CPU) usando un quad texturizado (GPU)
            self.upload_cpu_texture_changed()
            self.draw_textured_fullscreen_quad()
            return

        # ----------------------------------------------------
        # 2. Pipeline Z-Buffer (GPU)
        # ----------------------------------------------------
        if self.use_gpu_depth:
            glEnable(GL_DEPTH_TEST) # Z-Buffer activo
        else:
            glDisable(GL_DEPTH_TEST) # Sin ocultamiento

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Configuración de matrices (cámara, proyección, rotación)
        proj = mat_perspective(45.0, self.width / self.height, 0.1, 100.0)
        view = mat_lookat(np.array([0.0,2.0,self.distance], dtype=np.float32),
                          np.array([0.0,0.0,0.0], dtype=np.float32),
                          np.array([0.0,1.0,0.0], dtype=np.float32))
        view = view @ mat_rotation_x(np.radians(self.rot_x))
        view = view @ mat_rotation_y(np.radians(self.rot_y))
        
        model_plane = mat_identity()
        model_cube = mat_translation(0.0, 1.0, 0.0)

        program = self.prog_phong if self.use_phong else self.prog_gouraud
        glUseProgram(program)

        def set_mat4(name, m):
            loc = glGetUniformLocation(program, name)
            if loc != -1:
                # Se transpone la matriz numpy (row-major) para enviarla a OpenGL (column-major)
                glUniformMatrix4fv(loc, 1, GL_FALSE, m.astype(np.float32).T)

        set_mat4("view", view)
        set_mat4("projection", proj)

        # Configuración de uniformes de iluminación
        loc = glGetUniformLocation(program, "lightPos")
        if loc != -1: glUniform3fv(loc, 1, self.light_pos)
        loc = glGetUniformLocation(program, "viewPos")
        if loc != -1: glUniform3fv(loc, 1, np.array([0.0,2.0,self.distance], dtype=np.float32))
        loc = glGetUniformLocation(program, "lightColor")
        if loc != -1: glUniform3fv(loc, 1, self.light_color)
        loc = glGetUniformLocation(program, "shininess")
        if loc != -1: glUniform1f(loc, self.shininess)

        # draw plane
        set_mat4("model", model_plane)
        glBindVertexArray(self.plane_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        # draw cube
        set_mat4("model", model_cube)
        glBindVertexArray(self.cube_vao)
        glDrawArrays(GL_TRIANGLES, 0, 36)

        glBindVertexArray(0)
        glUseProgram(0)

    # -----------------------
    # S-BUFFER implementation (scanline/CPU)
    # -----------------------
    def render_sbuffer_to_cpu(self):
        sh, sw = self.cpu_fb.shape[0], self.cpu_fb.shape[1]
        
        # Limpiar buffers
        self.cpu_fb.fill(0)
        self.sbuffer = [dict() for _ in range(sh)]
        self.sbuffer_changed_lines.clear()

        # Matrices de transformación (las mismas que en la GPU)
        proj = mat_perspective(45.0, self.width / self.height, 0.1, 100.0)
        view = mat_lookat(np.array([0.0,2.0,self.distance], dtype=np.float32),
                          np.array([0.0,0.0,0.0], dtype=np.float32),
                          np.array([0.0,1.0,0.0], dtype=np.float32))
        view = view @ mat_rotation_x(np.radians(self.rot_x))
        view = view @ mat_rotation_y(np.radians(self.rot_y))
        model_plane = mat_identity()
        model_cube = mat_translation(0.0, 1.0, 0.0)

        # helper: transforma vértices de geometría a espacio de clip/pantalla/profundidad
        def extract_tris(arr, model_mat):
            tris = []
            if arr is None or arr.size == 0: return tris
            nverts = arr.size // 9
            v = arr.reshape((nverts, 9))
            for i in range(0, nverts, 3):
                if i + 2 >= nverts: break
                verts = []
                for j in range(3):
                    pos = np.array([v[i+j,0], v[i+j,1], v[i+j,2], 1.0], dtype=np.float32)
                    nd = np.array([v[i+j,3], v[i+j,4], v[i+j,5]], dtype=np.float32)
                    col = np.array([v[i+j,6], v[i+j,7], v[i+j,8]], dtype=np.float32)
                    
                    world_pos = (model_mat @ pos)[:3]
                    mat3 = np.asarray(model_mat)[:3,:3].astype(np.float32)
                    world_norm = mat3 @ nd
                    
                    clip = proj @ (view @ (model_mat @ pos))
                    w = clip[3] if abs(clip[3]) > 1e-8 else 1e-8
                    ndc = clip[:3] / w
                    
                    screen = ndc_to_screen(ndc[:2], sw, sh)
                    depth = float((ndc[2]*0.5) + 0.5) # [0, 1] depth
                    
                    verts.append({
                        'pos_world': world_pos, 'normal_world': world_norm, 'color': col,
                        'ndc': ndc, 'screen': screen, 'depth': depth, 'clip_w': float(w)
                    })
                tris.append(verts)
            return tris

        tris_plane = extract_tris(self.plane, model_plane)
        tris_cube = extract_tris(self.cube, model_cube)
        all_tris = tris_plane + tris_cube

        # ---------------------------------
        # Rasterización de triángulos
        # ---------------------------------
        for tri in all_tris:
            a = tri[0]['screen']; b = tri[1]['screen']; c = tri[2]['screen']
            # Bounding box en pantalla
            minx = int(np.floor(min(a[0], b[0], c[0]))); maxx = int(np.ceil(max(a[0], b[0], c[0])))
            miny = int(np.floor(min(a[1], b[1], c[1]))); maxy = int(np.ceil(max(a[1], b[1], c[1])))
            minx = clamp(minx, 0, sw-1); maxx = clamp(maxx, 0, sw-1)
            miny = clamp(miny, 0, sh-1); maxy = clamp(maxy, 0, sh-1)

            # Pre-cálculo para interpolación perspectiva
            inv_ws = np.array([1.0 / max(1e-8, tri[i]['clip_w']) for i in range(3)], dtype=np.float32)
            attrs = []
            for i in range(3):
                p = tri[i]
                attrs.append({
                    'color_div_w': p['color'] * inv_ws[i],
                    'normal_div_w': p['normal_world'] * inv_ws[i],
                    'pos_div_w': p['pos_world'] * inv_ws[i],
                    'inv_w': inv_ws[i],
                    'ndc_z_div_w': p['ndc'][2] * inv_ws[i]
                })

            for py in range(miny, maxy+1):
                line_dict = self.sbuffer[py]
                for px in range(minx, maxx+1):
                    p = np.array([px + 0.5, py + 0.5], dtype=np.float32) # Centro del píxel
                    u,v,w_ = barycentric(p, a, b, c)
                    
                    # Prueba de pertenencia al triángulo
                    if u < -1e-6 or v < -1e-6 or w_ < -1e-6:
                        continue
                        
                    # Interpolación perspectiva de 1/W
                    invw_p = attrs[0]['inv_w']*u + attrs[1]['inv_w']*v + attrs[2]['inv_w']*w_
                    if invw_p == 0: continue
                    
                    # Profundidad corregida
                    depth = (attrs[0]['ndc_z_div_w']*u + attrs[1]['ndc_z_div_w']*v + attrs[2]['ndc_z_div_w']*w_) / invw_p
                    depth = float(depth)
                    
                    # S-Buffer / Z-Test: Si la profundidad almacenada es menor o igual, descartar
                    prev = line_dict.get(px, None)
                    if prev is not None and prev <= depth:
                        continue

                    # Interpolación perspectiva del color, normal y posición mundial
                    color = (attrs[0]['color_div_w']*u + attrs[1]['color_div_w']*v + attrs[2]['color_div_w']*w_) / invw_p
                    normal = (attrs[0]['normal_div_w']*u + attrs[1]['normal_div_w']*v + attrs[2]['normal_div_w']*w_) / invw_p
                    posw = (attrs[0]['pos_div_w']*u + attrs[1]['pos_div_w']*v + attrs[2]['pos_div_w']*w_) / invw_p

                    # Re-normalizar y Shading (modelo Phong en CPU)
                    nlen = np.linalg.norm(normal)
                    nrm = normal / nlen if nlen > 1e-6 else np.array([0.0,1.0,0.0])

                    ambient = 0.1 * self.light_color * color
                    L = self.light_pos - posw; L = L / (np.linalg.norm(L) + 1e-12)
                    diff = max(np.dot(nrm, L), 0.0)
                    diffuse = diff * self.light_color * color
                    V = np.array([0.0, 2.0, self.distance]) - posw; V = V / (np.linalg.norm(V) + 1e-12)
                    R = 2.0 * np.dot(nrm, L) * nrm - L
                    R = R / (np.linalg.norm(R) + 1e-12)
                    spec = pow(max(np.dot(R, V), 0.0), self.shininess)
                    specular = spec * self.light_color
                    
                    shaded = ambient + diffuse + specular
                    shaded_clamped = np.clip(shaded, 0.0, 1.0)
                    rgb8 = (shaded_clamped * 255.0).astype(np.uint8)

                    # Escribir en Frame Buffer y S-Buffer
                    self.cpu_fb[py, px, :] = rgb8
                    line_dict[px] = depth
                    self.sbuffer_changed_lines.add(py)

    # -----------------------
    # Upload & draw CPU texture
    # -----------------------
    def upload_cpu_texture_changed(self):
        # Transfiere el contenido del Frame Buffer (CPU) a una Textura (GPU) para su visualización
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        sw = self.cpu_fb.shape[1]; sh = self.cpu_fb.shape[0]
        
        # En esta demo, se sube todo el buffer para simplificar la lógica de SubImage
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, sw, sh, GL_RGB, GL_UNSIGNED_BYTE, self.cpu_fb.tobytes())
        
        glBindTexture(GL_TEXTURE_2D, 0)
        self.sbuffer_changed_lines.clear()

    def draw_textured_fullscreen_quad(self):
        # Dibuja un quad en 2D que mapea la textura (el resultado del render CPU) a toda la ventana
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0,1,0,1,-1,1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0,0.0); glVertex2f(0.0,0.0)
        glTexCoord2f(1.0,0.0); glVertex2f(1.0,0.0)
        glTexCoord2f(1.0,1.0); glVertex2f(1.0,1.0)
        glTexCoord2f(0.0,1.0); glVertex2f(0.0,1.0)
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0); glDisable(GL_TEXTURE_2D)
        glMatrixMode(GL_MODELVIEW); glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix()

    # mouse
    def mousePressEvent(self, event):
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.last_pos.x(); dy = event.y() - self.last_pos.y()
        self.rot_y += dx * 0.5; self.rot_x += dy * 0.5
        self.last_pos = event.pos()
        self.sbuffer_needs_update = True
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        self.distance -= delta * 0.5
        self.distance = max(2.0, min(self.distance, 40.0))
        self.sbuffer_needs_update = True
        self.update()

# -----------------------
# MAIN WINDOW
# -----------------------
class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Z-Buffer (GPU) vs S-Buffer (CPU) - Demo")
        self.glw = GLWidget()
        self.setCentralWidget(self.glw)

        dock = QtWidgets.QDockWidget("Panel de Control")
        box = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout()

        layout.addWidget(QtWidgets.QLabel("<b>Método de ocultamiento</b>"))
        rb_gpu = QtWidgets.QRadioButton("Z-Buffer (GPU)"); rb_gpu.setChecked(True)
        rb_cpu = QtWidgets.QRadioButton("S-Buffer (CPU)")
        
        # Conexión de los Radio Buttons
        rb_gpu.toggled.connect(lambda val: self.set_mode_gpu(val))
        rb_cpu.toggled.connect(lambda val: self.set_mode_sbuffer(val))
        layout.addWidget(rb_gpu); layout.addWidget(rb_cpu)

        cb_depth = QtWidgets.QCheckBox("Activar Z-Buffer GPU (glEnable(GL_DEPTH_TEST))"); cb_depth.setChecked(True)
        cb_depth.stateChanged.connect(lambda s: self.toggle_gpu_depth(s))
        layout.addWidget(cb_depth)

        cb_ph = QtWidgets.QCheckBox("Usar Phong (GPU). Si no, Gouraud"); cb_ph.setChecked(True)
        cb_ph.stateChanged.connect(lambda s: self.toggle_phong(s))
        layout.addWidget(cb_ph)

        layout.addSpacing(8)
        layout.addWidget(QtWidgets.QLabel("<b>Opciones S-Buffer</b>"))
        self.cb_half = QtWidgets.QCheckBox("Render S-Buffer a 0.5x (más rápido)")
        self.cb_half.stateChanged.connect(self.toggle_sbuffer_scale)
        layout.addWidget(self.cb_half)

        btn = QtWidgets.QPushButton("Forzar redraw"); btn.clicked.connect(self.force_redraw)
        layout.addWidget(btn)
        layout.addStretch()
        box.setLayout(layout); dock.setWidget(box)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def set_mode_gpu(self, val):
        if val:
            self.glw.use_sbuffer = False
            self.glw.sbuffer_needs_update = True
            self.glw.update()
    def set_mode_sbuffer(self, val):
        if val:
            self.glw.use_sbuffer = True
            glDisable(GL_DEPTH_TEST)
            self.glw.sbuffer_needs_update = True
            self.glw.update()
    def toggle_gpu_depth(self, state):
        self.glw.use_gpu_depth = bool(state); self.glw.update()
    def toggle_phong(self, state):
        self.glw.use_phong = bool(state); self.glw.update()
    def toggle_sbuffer_scale(self, state):
        self.glw.sbuffer_scale = 0.5 if state else 1.0
        # Forzar re-inicialización de buffers al cambiar la escala
        self.glw.resizeGL(self.glw.width, self.glw.height)
        self.glw.sbuffer_needs_update = True
        self.glw.update()
    def force_redraw(self):
        self.glw.sbuffer_needs_update = True
        self.glw.update()

# -----------------------
# ENTRYPOINT
# -----------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = Window(); w.resize(1100,700); w.show()
    sys.exit(app.exec_())
