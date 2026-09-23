import os
from stl import mesh
import numpy as np

def swap_y_z_axes(mesh_data):
    """
    Swaps the Y and Z axes of the given mesh.
    
    Parameters:
        mesh_data (mesh.Mesh): The mesh object to transform.
        
    Returns:
        mesh.Mesh: The transformed mesh object with Y and Z axes swapped.
    """
    # Swap Y and Z coordinates
    # Original coordinates: [x, y, z]
    # After swapping: [x, z, y]
    mesh_data.vectors = mesh_data.vectors[:, :, [0, 2, 1]]
    return mesh_data

def process_all_stl_files(directory):
    """
    Processes all STL files in the specified directory by swapping their Y and Z axes.
    
    Parameters:
        directory (str): The path to the directory containing STL files.
    """
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter STL files (case-insensitive)
    stl_files = [f for f in files if f.lower().endswith('.stl')]
    
    if not stl_files:
        print("No STL files found in the current directory.")
        return
    
    print(f"Found {len(stl_files)} STL file(s) to process.")
    
    for stl_file in stl_files:
        input_path = os.path.join(directory, stl_file)
        try:
            print(f"Processing '{stl_file}'...")
            # Load the STL file
            your_mesh = mesh.Mesh.from_file(input_path)
            
            # Swap Y and Z axes
            your_mesh = swap_y_z_axes(your_mesh)
            
            # Save the modified mesh back to the same file (overwrite)
            your_mesh.save(input_path)
            print(f"Successfully updated '{stl_file}'.")
        except Exception as e:
            print(f"Failed to process '{stl_file}': {e}")

if __name__ == "__main__":
    # Get the current working directory
    current_directory = os.getcwd()
    print(f"Current directory: {current_directory}")
    
    # Optional: Backup original STL files before overwriting
    backup = False  # Set to True to create backups
    
    if backup:
        backup_dir = os.path.join(current_directory, "backup_stl_files")
        os.makedirs(backup_dir, exist_ok=True)
        for file in os.listdir(current_directory):
            if file.lower().endswith('.stl'):
                src = os.path.join(current_directory, file)
                dst = os.path.join(backup_dir, file)
                try:
                    if not os.path.exists(dst):
                        with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                            fdst.write(fsrc.read())
                        print(f"Backed up '{file}' to '{backup_dir}'.")
                except Exception as e:
                    print(f"Failed to backup '{file}': {e}")
    
    # Process all STL files
    process_all_stl_files(current_directory)
    
    print("All STL files have been processed.")