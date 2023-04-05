import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *  # Make sure to import GLU for gluPerspective

from OpenGL.GLUT import *
from PIL import Image

def load_texture(texture_path):
    image = Image.open(texture_path)
    image_data = image.tobytes()
    width, height = image.size

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, width, height, GL_RGB, GL_UNSIGNED_BYTE, image_data)

    return texture_id

def draw_textured_plane(texture_id, width, height):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glBegin(GL_QUADS)
    
    glTexCoord2f(0, 0)
    glVertex3f(-width / 2, -height / 2, 0)

    glTexCoord2f(1, 0)
    glVertex3f(width / 2, -height / 2, 0)

    glTexCoord2f(1, 1)
    glVertex3f(width / 2, height / 2, 0)

    glTexCoord2f(0, 1)
    glVertex3f(-width / 2, height / 2, 0)

    glEnd()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    texture_path = "../data/DTU/scan24/image/000000.png"  # Provide the path to your texture image
    texture_id = load_texture(texture_path)

    glEnable(GL_TEXTURE_2D)

    angle_x, angle_y = 0, 0
    dragging = False
    last_mouse_x, last_mouse_y = None, None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    print(event)
                    last_mouse_x, last_mouse_y = event.pos
                    dragging = True

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    print(event)
                    dragging = False
            if event.type == pygame.MOUSEMOTION:
                if dragging:
                    mouse_dx = event.pos[0] - last_mouse_x
                    mouse_dy = event.pos[1] - last_mouse_y
                    angle_x += mouse_dy * 0.5
                    angle_y += mouse_dx * 0.5
                    last_mouse_x, last_mouse_y = event.pos
            if event.type == pygame.MOUSEWHEEL:
                print(event)
                if event.y > 0:
                    glTranslatef(0, 0, 0.5)
                else:
                    glTranslatef(0, 0, -0.5)
            
            if event.type == pygame.KEYDOWN:
                print(event, event.key)
                if event.key == 113: # q
                    pygame.quit()
                    quit()
                if event.key == 114: # r
                    glLoadIdentity()
                    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
                    glTranslatef(0.0, 0.0, -5)


            # if event.type == pygame.MOUSEMOTION:
                # print(event)

        glRotatef(angle_x, 1, 0, 0)
        glRotatef(angle_y, 0, 1, 0)
        angle_x = 0
        angle_y = 0

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        draw_textured_plane(texture_id, 5, 5)

        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()
