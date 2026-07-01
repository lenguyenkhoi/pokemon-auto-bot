import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "Pikachu-Matching-Game"))

# Import truc tiep PikachuEnv da gop tu pikachu.py
from pikachu import PikachuEnv