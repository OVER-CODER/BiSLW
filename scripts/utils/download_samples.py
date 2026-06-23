#!/usr/bin/env python3
"""Generate diverse sample images for qualitative results."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os

def create_landscape(size=512):
    """Create a synthetic landscape image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Sky gradient (top blue to light blue)
    for y in range(size // 2):
        ratio = y / (size // 2)
        img[y, :] = [int(135 + 80 * ratio), int(206 + 30 * ratio), int(235 + 20 * ratio)]
    
    # Mountains
    for x in range(size):
        # Multiple mountain peaks
        h1 = int(size * 0.4 + 50 * np.sin(x * 0.02) + 30 * np.sin(x * 0.05))
        h2 = int(size * 0.45 + 40 * np.sin(x * 0.03 + 1) + 20 * np.sin(x * 0.07))
        
        # Far mountains (lighter)
        for y in range(h1, size):
            if y < size * 0.6:
                img[y, x] = [100, 120, 140]
        
        # Near mountains (darker)
        for y in range(h2, size):
            if y < size * 0.7:
                img[y, x] = [60, 80, 100]
    
    # Grass/ground
    for y in range(int(size * 0.7), size):
        for x in range(size):
            noise = np.random.randint(-20, 20)
            img[y, x] = [34 + noise, 139 + noise, 34 + noise]
    
    return Image.fromarray(img)


def create_portrait(size=512):
    """Create a synthetic portrait-like image."""
    img = Image.new('RGB', (size, size), (240, 220, 200))
    draw = ImageDraw.Draw(img)
    
    # Background gradient
    for y in range(size):
        ratio = y / size
        color = (int(200 - 50 * ratio), int(180 - 40 * ratio), int(160 - 30 * ratio))
        draw.line([(0, y), (size, y)], fill=color)
    
    # Face oval
    cx, cy = size // 2, size // 2
    draw.ellipse([cx - 120, cy - 160, cx + 120, cy + 140], fill=(255, 220, 185))
    
    # Eyes
    draw.ellipse([cx - 60, cy - 40, cx - 30, cy - 10], fill=(255, 255, 255))
    draw.ellipse([cx + 30, cy - 40, cx + 60, cy - 10], fill=(255, 255, 255))
    draw.ellipse([cx - 52, cy - 35, cx - 38, cy - 15], fill=(70, 50, 30))
    draw.ellipse([cx + 38, cy - 35, cx + 52, cy - 15], fill=(70, 50, 30))
    
    # Nose and mouth
    draw.polygon([(cx, cy - 10), (cx - 15, cy + 40), (cx + 15, cy + 40)], fill=(245, 200, 170))
    draw.arc([cx - 40, cy + 60, cx + 40, cy + 100], 0, 180, fill=(180, 100, 100), width=3)
    
    # Hair
    draw.ellipse([cx - 130, cy - 200, cx + 130, cy - 100], fill=(50, 30, 20))
    
    return img.filter(ImageFilter.GaussianBlur(2))


def create_animal(size=512):
    """Create a synthetic animal (cat) image."""
    img = Image.new('RGB', (size, size), (200, 220, 200))
    draw = ImageDraw.Draw(img)
    
    # Background
    for y in range(size):
        for x in range(size):
            noise = np.random.randint(-10, 10)
            img.putpixel((x, y), (180 + noise, 200 + noise, 180 + noise))
    
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2 + 50
    
    # Body
    draw.ellipse([cx - 100, cy - 50, cx + 100, cy + 120], fill=(255, 180, 100))
    
    # Head
    draw.ellipse([cx - 80, cy - 150, cx + 80, cy], fill=(255, 190, 110))
    
    # Ears
    draw.polygon([(cx - 70, cy - 130), (cx - 90, cy - 200), (cx - 30, cy - 150)], fill=(255, 180, 100))
    draw.polygon([(cx + 70, cy - 130), (cx + 90, cy - 200), (cx + 30, cy - 150)], fill=(255, 180, 100))
    
    # Eyes
    draw.ellipse([cx - 50, cy - 100, cx - 20, cy - 60], fill=(100, 200, 100))
    draw.ellipse([cx + 20, cy - 100, cx + 50, cy - 60], fill=(100, 200, 100))
    draw.ellipse([cx - 40, cy - 90, cx - 30, cy - 70], fill=(0, 0, 0))
    draw.ellipse([cx + 30, cy - 90, cx + 40, cy - 70], fill=(0, 0, 0))
    
    # Nose
    draw.polygon([(cx, cy - 50), (cx - 15, cy - 30), (cx + 15, cy - 30)], fill=(255, 150, 150))
    
    # Whiskers
    for i in range(-2, 3):
        draw.line([(cx - 20, cy - 35 + i * 10), (cx - 80, cy - 45 + i * 15)], fill=(50, 50, 50), width=1)
        draw.line([(cx + 20, cy - 35 + i * 10), (cx + 80, cy - 45 + i * 15)], fill=(50, 50, 50), width=1)
    
    return img.filter(ImageFilter.GaussianBlur(1))


def create_urban(size=512):
    """Create a synthetic urban/city image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Night sky
    for y in range(size):
        img[y, :] = [max(0, 20 - y // 30), max(0, 30 - y // 25), max(0, 60 - y // 20)]
    
    # Add stars
    for _ in range(100):
        x, y = np.random.randint(0, size), np.random.randint(0, size // 2)
        img[y, x] = [200 + np.random.randint(55), 200 + np.random.randint(55), 200 + np.random.randint(55)]
    
    # Buildings
    for _ in range(15):
        bx = np.random.randint(0, size - 80)
        bw = np.random.randint(40, 100)
        bh = np.random.randint(150, 400)
        
        color = np.random.randint(40, 80)
        img[size - bh:size, bx:bx + bw] = [color, color, color + 20]
        
        # Windows
        for wy in range(size - bh + 10, size - 10, 25):
            for wx in range(bx + 8, bx + bw - 8, 15):
                if np.random.random() > 0.3:
                    window_color = [255, 255, 200] if np.random.random() > 0.5 else [255, 200, 100]
                    img[wy:wy + 15, wx:wx + 8] = window_color
    
    # Street
    img[size - 50:size, :] = [50, 50, 50]
    
    # Street lights
    for x in [100, 250, 400]:
        img[size - 60:size - 50, x:x + 5] = [200, 200, 150]
        for dy in range(20):
            for dx in range(-30, 30):
                if 0 <= x + dx < size:
                    alpha = max(0, 1 - abs(dx) / 30 - dy / 20)
                    img[size - 50 + dy, x + dx] = [int(255 * alpha), int(255 * alpha), int(200 * alpha)]
    
    return Image.fromarray(img)


def create_artwork(size=512):
    """Create a synthetic artwork (abstract/impressionist style)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Base warm colors
    for y in range(size):
        for x in range(size):
            img[y, x] = [
                int(200 + 55 * np.sin(x * 0.05 + y * 0.03)),
                int(150 + 50 * np.sin(x * 0.03 + y * 0.05)),
                int(100 + 50 * np.sin(x * 0.07 + y * 0.02))
            ]
    
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Brush strokes
    colors = [(255, 200, 50), (255, 100, 50), (200, 50, 50), (255, 255, 100), (150, 200, 255)]
    for _ in range(200):
        x1 = np.random.randint(0, size)
        y1 = np.random.randint(0, size)
        x2 = x1 + np.random.randint(-50, 50)
        y2 = y1 + np.random.randint(-50, 50)
        color = colors[np.random.randint(len(colors))]
        width = np.random.randint(5, 20)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    
    # Add some circular elements (like sunflowers)
    for _ in range(5):
        cx = np.random.randint(100, size - 100)
        cy = np.random.randint(100, size - 100)
        r = np.random.randint(30, 60)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(100, 80, 50))
        
        # Petals
        for angle in range(0, 360, 30):
            px = cx + int(r * 1.5 * np.cos(np.radians(angle)))
            py = cy + int(r * 1.5 * np.sin(np.radians(angle)))
            draw.ellipse([px - 15, py - 15, px + 15, py + 15], fill=(255, 220, 50))
    
    return pil_img.filter(ImageFilter.GaussianBlur(2))


def create_render(size=512):
    """Create a synthetic 3D render style image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Gradient background
    for y in range(size):
        for x in range(size):
            img[y, x] = [
                int(50 + 30 * (y / size)),
                int(50 + 40 * (y / size)),
                int(80 + 50 * (y / size))
            ]
    
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Ground plane
    for y in range(size // 2, size):
        alpha = (y - size // 2) / (size // 2)
        color = (int(100 * alpha), int(100 * alpha), int(120 * alpha))
        draw.line([(0, y), (size, y)], fill=color)
    
    # Grid lines
    for i in range(0, size, 40):
        # Horizontal
        y = size // 2 + int(i * 0.8)
        if y < size:
            draw.line([(0, y), (size, y)], fill=(80, 80, 100), width=1)
        # Vertical perspective
        x1, x2 = i, size - i
        draw.line([(x1, size // 2), (i // 2, size)], fill=(80, 80, 100), width=1)
        draw.line([(x2, size // 2), (size - i // 2, size)], fill=(80, 80, 100), width=1)
    
    # Sphere
    cx, cy = size // 2, size // 3
    r = 80
    for dy in range(-r, r):
        for dx in range(-r, r):
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist < r:
                # Shading
                light_x, light_y = -0.5, -0.5
                normal_x = dx / r
                normal_y = dy / r
                normal_z = np.sqrt(max(0, 1 - normal_x ** 2 - normal_y ** 2))
                
                dot = max(0, normal_x * light_x + normal_y * light_y + normal_z * 0.7)
                intensity = 0.3 + 0.7 * dot
                
                color = (int(200 * intensity), int(50 * intensity), int(50 * intensity))
                pil_img.putpixel((cx + dx, cy + dy), color)
    
    # Cube
    cx, cy = size // 4, size // 2
    s = 60
    # Front face
    draw.polygon([(cx, cy), (cx + s, cy), (cx + s, cy + s), (cx, cy + s)], fill=(50, 150, 50))
    # Top face
    draw.polygon([(cx, cy), (cx + s // 2, cy - s // 2), (cx + s + s // 2, cy - s // 2), (cx + s, cy)], fill=(100, 200, 100))
    # Side face
    draw.polygon([(cx + s, cy), (cx + s + s // 2, cy - s // 2), (cx + s + s // 2, cy + s // 2), (cx + s, cy + s)], fill=(30, 100, 30))
    
    return pil_img


def main():
    size = 512
    os.makedirs('sample_images', exist_ok=True)
    
    generators = [
        ('landscape.jpg', create_landscape),
        ('portrait.jpg', create_portrait),
        ('animal.jpg', create_animal),
        ('urban.jpg', create_urban),
        ('artwork.jpg', create_artwork),
        ('render.jpg', create_render),
    ]
    
    for filename, gen_func in generators:
        filepath = os.path.join('sample_images', filename)
        if not os.path.exists(filepath):
            print(f'Generating {filename}...')
            img = gen_func(size)
            img.save(filepath, quality=95)
            print(f'  Saved to {filepath}')
        else:
            print(f'{filename} already exists')
    
    print('Done!')


if __name__ == '__main__':
    main()
