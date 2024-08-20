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

from os.path import join, dirname, basename, abspath, normpath
from glob import glob

def get_sound_files(database_name):
    database_path = dirname(database_name+'/')

    all_path = []
    extensions = ['*.wav', '*.mp3', '*.aiff']
    for ext in extensions:
        all_path.extend(glob(join(database_path+'/*/', ext)))

    all_names = []
    for path in range(len(all_path)):
        all_names.append(basename(all_path[path]))
    
    print(f'{database_name} : {len(all_path)} files found')
    return all_path, all_names