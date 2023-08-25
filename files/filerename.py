import os

folder_path = 'C:\\Users\\IT\\AppData\\Local\\Programs\\Python\\Python310\\venv4wxpython\\Scripts\\face recognition\\datasetnew\\yourfoldername'  # Replace with the actual folder path
file_extension = '.jpg'  # Replace with the desired file extension

# Get a list of all files in the folder with the specified extension
files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
print("here")
# Sort the list of files alphabetically
files.sort()
print("sorted")
# Rename the files with sequential numbering
for i, filename in enumerate(files, start=1):
    new_filename = f'yourspecificname{i}{file_extension}'
    print(new_filename)
    old_filepath = os.path.join(folder_path, filename)
    new_filepath = os.path.join(folder_path, new_filename)
    os.rename(old_filepath, new_filepath)
    print(f'Renamed {filename} to {new_filename}')
