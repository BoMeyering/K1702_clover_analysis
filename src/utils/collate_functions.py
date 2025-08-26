"""
src/utils/collate_functions.py
"""

import torch

def custom_collate(batch):
    return tuple(zip(*batch))