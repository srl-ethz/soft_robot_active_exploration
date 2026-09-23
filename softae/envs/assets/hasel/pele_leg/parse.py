import os
import xml.etree.ElementTree as ET

# Parse the existing XML file
tree = ET.parse('leg_setup.xml')
root = tree.getroot()

# Find the assets section
assets = root.find('assets')

# If assets section doesn't exist, create it
if assets is None:
    assets = ET.SubElement(root, 'assets')

# Iterate through all files in the assets folder
for filename in os.listdir('assets'):
    if filename.lower().endswith('.stl'):
        # Create a new mesh element
        mesh = ET.SubElement(assets, 'mesh')
        
        # Set the file attribute to the STL filename
        mesh.set('file', filename)
        
        # Set the name attribute to the filename without extension
        name = os.path.splitext(filename)[0]
        mesh.set('name', name)

# Write the updated XML back to the file
tree.write('leg_setup.xml', encoding='utf-8', xml_declaration=True)