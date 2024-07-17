import csv
import os
import random
import webcolors
from PIL import Image
import matplotlib.colors as mcolors
import trimesh
import numpy as np
import ShapeGenerator2D
colors = mcolors.CSS4_COLORS

def random_color_tuple():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def rgb_to_hex(rgb_tuple):
    return "#{:02x}{:02x}{:02x}".format(rgb_tuple[0], rgb_tuple[1], rgb_tuple[2])

def closest_color(requested_color):
    min_distance = float('inf')
    closest_name = None
    for name, hex_color in colors.items():
        r_c, g_c, b_c = mcolors.hex2color(hex_color)
        r_c, g_c, b_c = int(r_c * 255), int(g_c * 255), int(b_c * 255)
        distance = ((r_c - requested_color[0]) ** 2 +
                    (g_c - requested_color[1]) ** 2 +
                    (b_c - requested_color[2]) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    return closest_name

def get_color_name(rgb_tuple):
    try:
        hex_color = rgb_to_hex(rgb_tuple)
        color_name = webcolors.hex_to_name(hex_color)
    except ValueError:
        color_name = closest_color(rgb_tuple)
    return color_name

def create_texture_image(texture_filepath, shape_color_RGB):
    texture_size = 256
    img = Image.new('RGB', (texture_size, texture_size), shape_color_RGB)
    img.save(texture_filepath)

def draw_pyramid(base_length, height, obj_filepath, texture_filepath, shape_color_RGB):
    vertices = np.array([
        [0, 0, 0],
        [base_length, 0, 0],
        [base_length, base_length, 0],
        [0, base_length, 0],
        [base_length / 2, base_length / 2, height]
    ])

    faces = np.array([
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4]
    ])

    pyramid = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Generate UV coordinates
    uv_coords = np.array([
        [0, 0],  # UV for vertex 0
        [1, 0],  # UV for vertex 1
        [1, 1],  # UV for vertex 2
        [0, 1],  # UV for vertex 3
        [0.5, 0.5]  # UV for vertex 4 (apex)
    ])

    pyramid.visual = trimesh.visual.TextureVisuals(uv=uv_coords, image=texture_filepath)
    pyramid.export(obj_filepath)


def draw_cube(side_length, obj_filepath, texture_filepath, shape_color_RGB):
    vertices = np.array([
        [0, 0, 0],
        [side_length, 0, 0],
        [side_length, side_length, 0],
        [0, side_length, 0],
        [0, 0, side_length],
        [side_length, 0, side_length],
        [side_length, side_length, side_length],
        [0, side_length, side_length]
    ])

    faces = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 3, 7, 4]
    ])

    cube = trimesh.Trimesh(vertices=vertices, faces=faces)
    uv_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    cube.visual = trimesh.visual.TextureVisuals(uv=np.tile(uv_coords, (6, 1)), image=texture_filepath)
    cube.export(obj_filepath)

def draw_sphere(diameter, obj_filepath, texture_filepath, shape_color_RGB):
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=diameter / 2)
    
    # Generate UV coordinates manually
    uvs = []
    for vertex in sphere.vertices:
        x, y, z = vertex
        u = 0.5 + (np.arctan2(z, x) / (2 * np.pi))
        v = 0.5 - (np.arcsin(y) / np.pi)
        uvs.append([u, v])
    uv_coords = np.array(uvs)
    
    sphere.visual = trimesh.visual.TextureVisuals(uv=uv_coords, image=texture_filepath)
    sphere.export(obj_filepath)


def save_to_csv(img_filepath, csv_filepath, shape_color_name, shape_type):
    file_existence = os.path.isfile(csv_filepath)

    with open(csv_filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_existence or os.stat(csv_filepath).st_size == 0:
            writer.writerow(['Image file name', 'Shape', 'Color name'])

        writer.writerow([img_filepath, shape_type, shape_color_name])

def model_generator(mdl_num):
    n = mdl_num
    output_dir1 = 'shapes3d'
    output_dir2 = 'shapes2d'
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)
    csv_filepath = 'shape_colors.csv'
    

    for i in range(n):
        shape_color_RGB = random_color_tuple()
        base_length, height = random.randint(50, 200), random.randint(50, 100)
        side_length = random.randint(50, 200)
        diameter = random.randint(50, 200)

        texture_filepath = os.path.join(output_dir2, f"texture_{i}.png")
        create_texture_image(texture_filepath, shape_color_RGB)

        shape_color_name = get_color_name(shape_color_RGB)
        
        # Draw and save pyramid
        png_filepath = os.path.join(output_dir2, f"triangle_{i}.png")
        ShapeGenerator2D.triangle_generator(base_length,height, png_filepath, shape_color_RGB)
        ShapeGenerator2D.save_to_csv(png_filepath, csv_filepath, shape_color_name, 'triangle')

        obj_filepath = os.path.join(output_dir1, f"pyramid_{i}.obj")
        draw_pyramid(base_length, height, obj_filepath, texture_filepath, shape_color_RGB)
        save_to_csv(obj_filepath, csv_filepath, shape_color_name, 'pyramid')
        
        # Draw and save cube
        png_filepath = os.path.join(output_dir2, f"square_{i}.png")
        ShapeGenerator2D.square_generator(side_length, png_filepath, shape_color_RGB)
        ShapeGenerator2D.save_to_csv(png_filepath, csv_filepath, shape_color_name, 'square')

        obj_filepath = os.path.join(output_dir1, f"cube_{i}.obj")
        draw_cube(side_length, obj_filepath, texture_filepath, shape_color_RGB)
        save_to_csv(obj_filepath, csv_filepath, shape_color_name, 'cube')
        
        # Draw and save sphere
        png_filepath = os.path.join(output_dir2, f"circle_{i}.png")
        ShapeGenerator2D.circle_generator(diameter, png_filepath, shape_color_RGB)
        ShapeGenerator2D.save_to_csv(png_filepath, csv_filepath, shape_color_name, 'circle')

        obj_filepath = os.path.join(output_dir1, f"sphere_{i}.obj")
        draw_sphere(diameter, obj_filepath, texture_filepath, shape_color_RGB)
        save_to_csv(obj_filepath, csv_filepath, shape_color_name, 'sphere')

model_generator(100)
