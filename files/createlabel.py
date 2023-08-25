import os

dataset_dir = 'C:\\Users\\IT\\AppData\\Local\\Programs\\Python\\Python310\\venv4wxpython\\Scripts\\face recognition\\datasetnew'
label_file = 'label_file.txt'

with open(label_file, 'w') as file:
    for label, person_name in enumerate(os.listdir(dataset_dir)):
        person_dir = os.path.join(dataset_dir, person_name)
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            file.write(f'{image_path},{label}\n')
