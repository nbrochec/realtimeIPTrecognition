#############################################################################
# augmentations.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# GNU General Public License v3.0
#############################################################################
# Code description:
# Implement data augmentation methods
#############################################################################

import torch
import torch_audiomentations

'''
Principally using torch_audiomentations because:
1. Augmentations are generated inside the training loop.
2. torch_audiomentations provide cuda support enabling GPU computation.
'''

