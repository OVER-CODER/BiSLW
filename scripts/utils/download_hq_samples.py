#!/usr/bin/env python3
"""Download high-quality sample images from the internet."""

import urllib.request
import os
import ssl

# Bypass SSL verification for downloads
ssl._create_default_https_context = ssl._create_unverified_context

# High-quality sample images from Pexels (free to use)
urls = [
    # Busy street
    ("https://images.pexels.com/photos/1714208/pexels-photo-1714208.jpeg?auto=compress&cs=tinysrgb&w=800", "hq_busy_street.jpg"),
    # Graphics design / abstract art
    ("https://images.pexels.com/photos/2110951/pexels-photo-2110951.jpeg?auto=compress&cs=tinysrgb&w=800", "hq_graphics.jpg"),
    # Night city
    ("https://images.pexels.com/photos/1519088/pexels-photo-1519088.jpeg?auto=compress&cs=tinysrgb&w=800", "hq_night.jpg"),
    # Park
    ("https://images.pexels.com/photos/1770809/pexels-photo-1770809.jpeg?auto=compress&cs=tinysrgb&w=800", "hq_park.jpg"),
    # Crowded market street
    ("https://images.pexels.com/photos/3889843/pexels-photo-3889843.jpeg?auto=compress&cs=tinysrgb&w=800", "hq_market.jpg"),
    # Neon lights night
    ("https://images.pexels.com/photos/2526105/pexels-photo-2526105.jpeg?auto=compress&cs=tinysrgb&w=800", "hq_neon.jpg"),
    # Colorful design
    ("https://images.pexels.com/photos/1762851/pexels-photo-1762851.jpeg?auto=compress&cs=tinysrgb&w=800", "hq_colorful.jpg"),
    # Garden/nature park
    ("https://images.pexels.com/photos/158028/bellingrath-gardens-702702-702703-702700-158028.jpeg?auto=compress&cs=tinysrgb&w=800", "hq_garden.jpg"),
]

os.makedirs('sample_images/hq', exist_ok=True)

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}

for url, filename in urls:
    filepath = os.path.join('sample_images/hq', filename)
    if not os.path.exists(filepath):
        print(f'Downloading {filename}...')
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                with open(filepath, 'wb') as f:
                    f.write(response.read())
            print(f'  Saved to {filepath}')
        except Exception as e:
            print(f'  Failed: {e}')
    else:
        print(f'{filename} already exists')

print('Done!')
