# SRG268
Special Research Group 268

Anaconda commands to ensure accurate dependencies:
1. `conda activate sam_env`

Default imports:
```py
import os
import random
import sys
import cv2
import easyocr
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.stats import mode
import streamlit as st
from streamlit.web import cli as stcli
```