import os

def remove_folder_contents(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                remove_folder_contents(file_path)
                os.rmdir(file_path)
        except Exception as e:
            print(e)

#remove_folder_contents("./models")
            
def remove_specific_seed_files(folder, seed):
    for the_file in os.listdir(folder):
        if f"seed_{str(seed)}" in the_file:
            file_path = os.path.join(folder, the_file)
            os.unlink(file_path)

#remove_specific_seed_files("./models", 111)

"""
for the_file in os.listdir("./models"):
    if f"seed_{111}" in the_file:
        print(the_file)
"""
