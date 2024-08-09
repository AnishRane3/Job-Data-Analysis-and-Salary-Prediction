# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 19:36:11 2024

@author: ranea
"""

from bs4 import BeautifulSoup

# Load the HTML file
with open('EDA.html', 'r', encoding='utf-8') as file:
    soup = BeautifulSoup(file, 'lxml')

# Find all image tags
images = soup.find_all('img')

# Create a new HTML file with only images
with open('EDA_images.html', 'w', encoding='utf-8') as output_file:
    output_file.write('<html><head><title>EDA Images</title></head><body>\n')
    for img in images:
        output_file.write(str(img) + '\n')
    output_file.write('</body></html>')
