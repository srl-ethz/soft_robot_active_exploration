import os
import datetime

def rename_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.startswith("__Sim"):
            new_filename = filename[10:]
            new_path = os.path.join(folder_path, new_filename)

            # Check if the new file name already exists
            if os.path.exists(new_path):
                # Append a timestamp to the new file name to avoid collision
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                name, ext = os.path.splitext(new_filename)
                new_filename = f"{name}_{timestamp}{ext}"
                new_path = os.path.join(folder_path, new_filename)

            os.rename(os.path.join(folder_path, filename), new_path)
    
    return "Files renamed successfully."

# Example usage (you need to provide the actual folder path)
folder_path = "C:\\Users\\amrkz\\Downloads\\mujoco-leg-sim\\Leg_setup\\assets"
rename_files_in_folder(folder_path)
