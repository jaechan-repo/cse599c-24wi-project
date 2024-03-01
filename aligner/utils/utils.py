import os
import fnmatch

def find_file(directory, extension):
    for root, _, files in os.walk(directory):
        for file in files:
            if fnmatch.fnmatch(file, f'*.{extension}'):
                return os.path.join(root, file)
    raise FileNotFoundError