#############################################################################
# utils.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# MIT license 2024
#############################################################################
# Code description:
# Utilities defition
#############################################################################

from os.path import join, dirname, basename, abspath, normpath, isdir
from os import listdir
from glob import glob

def browse_subfolders(folder):
    subfolder_list = []
    
    for folder_name in listdir(folder):
        path = join(folder, folder_name)
        if isdir(path):
            subfolder_list.append(folder_name)
            sous_dossiers = browse_subfolders(path) 
            subfolder_list.extend(sous_dossiers)
        
    subfolder_list.sort()
    return subfolder_list

def get_sound_files(datafolderpath):
    database_path = dirname(datafolderpath+'/')

    all_path = []
    extensions = ['*.wav', '*.mp3', '*.aiff']
    for ext in extensions:
        all_path.extend(glob(join(database_path+'/*/', ext)))

    all_names = []
    for path in range(len(all_path)):
        all_names.append(basename(all_path[path]))
    
    print(f'{len(all_path)} files found')
    return all_path, all_names