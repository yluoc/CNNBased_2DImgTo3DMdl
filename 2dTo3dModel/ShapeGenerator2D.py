import csv
import os
import random
import webcolors
from PIL import Image, ImageDraw
import matplotlib.colors as mcolors

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
                    (b_c - requested_color[2]) ** 2) ** 0.5  # Use sqrt to ensure distance is a real number
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

def triangle_generator(base_length, height, img_filepath, shape_color_RGB):
    base_length = int(base_length)
    height = int(height)
    shape_color_name = get_color_name(shape_color_RGB)
    img = Image.new('RGBA', (base_length, height), (255, 255, 255, 0))
    img_draw = ImageDraw.Draw(img)
    triangle = [(0, height), (base_length, height), (base_length / 2, 0)]
    img_draw.polygon(triangle, outline=None, fill=shape_color_RGB)
    img.save(img_filepath, 'PNG')

    return shape_color_name

def square_generator(side_length, img_filepath, shape_color_RGB):
    side_length = int(side_length)
    shape_color_name = get_color_name(shape_color_RGB)
    img = Image.new('RGBA', (side_length, side_length), (255, 255, 255, 0))
    img_draw = ImageDraw.Draw(img)
    square = [(0, 0), (side_length, 0), (side_length, side_length), (0, side_length)]
    img_draw.polygon(square, outline=None, fill=shape_color_RGB)
    img.save(img_filepath, 'PNG')

    return shape_color_name

def circle_generator(diameter, img_filepath, shape_color_RGB):
    diameter = int(diameter)
    shape_color_name = get_color_name(shape_color_RGB)
    img = Image.new('RGBA', (diameter, diameter), (255, 255, 255, 0))
    img_draw = ImageDraw.Draw(img)
    img_draw.ellipse([(0, 0), (diameter, diameter)], outline=None, fill=shape_color_RGB)
    img.save(img_filepath, 'PNG')

    return shape_color_name

def save_to_csv(img_filepath, csv_filepath, shape_color_name, shape_type):
    file_existence = os.path.isfile(csv_filepath)

    with open(csv_filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_existence or os.stat(csv_filepath).st_size == 0:
            writer.writerow(['Image file name', 'Shape', 'Color name'])

        writer.writerow([img_filepath, shape_type, shape_color_name])

def image_generator(img_num):
    n = img_num
    output_dir = 'shapes2d'
    os.makedirs(output_dir, exist_ok=True)
    csv_filepath = os.path.join(output_dir, 'shape_colors.csv')

    for i in range(n):
        shape_color_RGB = random_color_tuple()

        base_length, height = random.randint(50, 200), random.randint(50, 100)
        side_length = random.randint(50, 200)
        diameter = random.randint(50, 200)

        img_filepath1 = os.path.join(output_dir, f"triangle_{i}.png")
        shape_color_name1 = triangle_generator(base_length, height, img_filepath1, shape_color_RGB)
    
        img_filepath2 = os.path.join(output_dir, f"square_{i}.png")
        shape_color_name2 = square_generator(side_length, img_filepath2, shape_color_RGB)
        
        img_filepath3 = os.path.join(output_dir, f"circle_{i}.png")
        shape_color_name3 = circle_generator(diameter, img_filepath3, shape_color_RGB)
        
        save_to_csv(img_filepath1, csv_filepath, shape_color_name1, "triangle")
        save_to_csv(img_filepath2, csv_filepath, shape_color_name2, "square")
        save_to_csv(img_filepath3, csv_filepath, shape_color_name3, "circle")

