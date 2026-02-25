Z-Buffer (GPU) vs S-Buffer (CPU) es una aplicación interactiva desarrollada en Python que compara dos técnicas de ocultamiento de superficies en gráficos por computadora:

  Z-Buffer implementado en GPU mediante OpenGL

 S-Buffer implementado en CPU mediante rasterización por software

El proyecto permite visualizar y analizar las diferencias de rendimiento, precisión y funcionamiento entre ambos enfoques dentro de una escena 3D iluminada con modelos de sombreado Phong y Gouraud.

La aplicación está construida con PyQt5, PyOpenGL y NumPy, e incluye interacción por mouse para rotación y zoom de la cámara.

 Características principales

Comparación directa entre Z-Buffer (GPU) y S-Buffer (CPU)

Implementación de shading Phong y Gouraud

Rasterización por software con interpolación perspectiva-correcta

Manejo manual de profundidad en CPU

Visualización en tiempo real

Control interactivo de cámara (rotación y zoom)

Escalado de resolución del S-Buffer para pruebas de rendimiento

 Tecnologías utilizadas

Python 3

PyQt5

PyOpenGL

NumPy

OpenGL 3.3 (Shaders GLSL)

 Ejecución
pip install PyQt5 PyOpenGL numpy
python MetodoS_Buffer_final.py
 Autor
José David Ariza
