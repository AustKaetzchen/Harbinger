# Harbinger.Segmentation
Developed by **Special Research Group 268, Confoederatio Research Division** (SRG268-CRD).

Requirements:
1. Ensure Anaconda is installed such that you have access to Anaconda Prompt:
2. Run Anaconda As Administrator > `conda init cmd.exe` to tie it to your Command Prompt system.
3. Restart Command Prompt (Administrator)
4. `conda env create -f environment.yml` (Installs all Python dependencies)
5. `conda activate sam_env` (Ensures accurate dependencies once installed)

The root app bundle is located in `./main/app.py`.

Default imports:
```py
import argparse
import json
import math
import os
import random
import sys
import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.stats import mode
import streamlit as st
from streamlit.web import cli as stcli
```
Imports are not fully minimised from sam_env, so sam_env may be bigger than needed. This should be checked by Confoederatio developers in the future.