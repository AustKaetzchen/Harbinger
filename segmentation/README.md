# Harbinger.Segmentation

Developed by **Special Research Group 268, Confoederatio Research Division** (SRG268-CRD). Functions as a performant alternative to Segment-Anything, especially for symbolic maps. Additionally applicable to real-world photos, graphics, and other use-cases.

Originally developed to aid in auto-vectorisation for [Naissance HGIS](https://github.com/ConfoederatioVF/Naissance).

## Installation:

Requirements:
1. Ensure Anaconda is installed such that you have access to Anaconda Prompt:
2. Run Anaconda As Administrator > `conda init cmd.exe` to tie it to your Command Prompt system.
3. Restart Command Prompt (Administrator)
4. `conda env create -f environment.yml` (Installs all Python dependencies)
5. `conda activate sam_env` (Ensures accurate dependencies once installed)

The root app bundle is located in `./main/app.py`. For developers, **Spyder** configuration files are already specified.

## CLI Usage.

Harbinger.Segmentation is designed for use with CLI workflows. By default, all that is required is requisite input pathing, i.e. `python app.py -i input.png`. It is also possible to specify a custom output folder: `python app.py -i input.png -o ../custom_output/`. See the below table for a full list of options and flags:

**Available CLI Arguments:**

<table>
  <tr>
    <th>Flag</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><kbd>-i</kbd>, <kbd>--input</kbd></td>
    <td>string</td>
    <td>None</td>
    <td>Path to input map image file.</td>
  </tr>
  <tr>
    <td><kbd>-o</kbd>, <kbd>--output-dir</kbd></td>
    <td>string</td>
    <td>output</td>
    <td>Directory where output files will be written.</td>
  </tr>
  <tr>
    <td><kbd>--gui</kbd></td>
    <td>flag</td>
    <td>False</td>
    <td>Explicitly launch Streamlit GUI in browser.</td>
  </tr>
  <tr>
    <td><kbd>--mask-legends</kbd> / <kbd>--no-mask-legends</kbd></td>
    <td>bool</td>
    <td>True</td>
    <td>Automatically detect and mask legend infoboxes.</td>
  </tr>
  <tr>
    <td><kbd>--save-semantic-features</kbd> / <kbd>--no-save-semantic-features</kbd></td>
    <td>bool</td>
    <td>True</td>
    <td>Export JSON labels file.</td>
  </tr>
  <tr>
    <td><kbd>--save-output-images</kbd> / <kbd>--no-save-output-images</kbd></td>
    <td>bool</td>
    <td>True</td>
    <td>Save all intermediate stage PNG images.</td>
  </tr>
  <tr>
    <td><kbd>--colour-thresh</kbd></td>
    <td>int</td>
    <td>15</td>
    <td>Colour similarity quantisation threshold (1-50).</td>
  </tr>
  <tr>
    <td><kbd>--edge-thresh</kbd></td>
    <td>int</td>
    <td>20</td>
    <td>Edge gradient threshold (1-50).</td>
  </tr>
  <tr>
    <td><kbd>--density-seeding-thresh</kbd></td>
    <td>int</td>
    <td>25</td>
    <td>Density seeding threshold (0-100).</td>
  </tr>
  <tr>
    <td><kbd>--border-buffer</kbd></td>
    <td>int</td>
    <td>1</td>
    <td>Border buffer padding size (1-8).</td>
  </tr>
  <tr>
    <td><kbd>--text-buffer</kbd></td>
    <td>int</td>
    <td>8</td>
    <td>OCR text mask padding size (1-50).</td>
  </tr>
  <tr>
    <td><kbd>--locales</kbd></td>
    <td>list</td>
    <td>en ru</td>
    <td>Language code list for EasyOCR.</td>
  </tr>
</table>

**Output Files.**

> [!NOTE]
> Unlike in the Streamlit GUI, these files do not undergo JPEG compression, and are presented iN the image's original resolution.

As the pipeline gradually completes, images in the chosen output folder are overriden one-by-one. These images correspond as follows:

`output/`:
- `1.png`: Original Map
- `2.png`: UI Masking
- `3.png`: Denoised Image
- `4.png`: Sharpness Layer
- `5.png`: Semantic Features
- `6.png`: Denoised Edges
- `7.png`: 1st-pass Segmentation
- `8.png`: 1st-pass Filtering
- `9.png`: Edge Restoration
- `10.png`: 2nd-pass Segmentation
- `11.png`: 1st-pass kNN Repair
- `12.png`: 2nd-pass kNN Repair

<ins>Semantic features</ins> are available as `output/labels.json`. They have a JSON contract as follows:

Object[], where each [n] is:
- `.name`: string
- `.centre`: [int, int] - The X, Y coordinates of the centre of the bounding box.
- `.extent`: [[int, int], [int, int], [int, int], [int, int]] - Representing the NW, NE, SW, and SE corners of the OCR bbox respectively.
- `.probability`: float - The certainty estimate with which this OCR label was read.

Non-Latin characters use Unicode display characters (i.e. \u0417).

**All available locales:**

```py
available_locales = [
  "abq", "ady", "af", "sq", "ang", "ar", "as", "ava", "az", "be",
  "bn", "bho", "bh", "bs", "bg", "che", "ch_sim", "ch_tra", "hr", "cs",
  "da", "dar", "nl", "en", "et", "fr", "de", "gom", "hi", "hu",
  "is", "id", "inh", "ga", "it", "ja", "kbd", "kn", "ko", "ku",
  "lbe", "la", "lv", "lez", "lt", "mah", "mai", "ms", "mt", "mi",
  "mr", "mn", "sck", "ne", "new", "no", "oc", "pi", "fa", "pl",
  "pt", "ro", "ru", "rs_cyrillic", "rs_latin", "sk", "sl", "es", "sw",
  "sv", "tab", "tl", "tjk", "ta", "te", "th", "tr", "uk", "ur",
  "ug", "uz", "vi", "cy"
]
```

Multiple locales can be selected at a time, but some combinations may be incompatible. You will be warned of these combinations where they appear, and it will fallback to English.

## GUI Usage.

The GUI for Harbinger.Segmentation relies on Streamlit, and can be served either over [Spyder](https://www.spyder-ide.org/) by running `app.py`, or through your terminal of choice: `python app.py --gui`. Options can be configured from the left sidebar, and execution will only take place when pressing 'Run'.

<img src = "https://i.postimg.cc/fLWh5L6F/61-harbinger-segmentation.png" width = "100%">
<div align = "center">An example of the Streamlit interface.</div>

## Other Notes.

The GUI is beneficial for first-time users. Note that if your image is highly noisy in the background, we recommend high Edge Gradient Thresholds - the maximum value is often recommended. Real-world photos should also use medium-high Colour Similarity Thresholds (~30-45). Additionally, if you still see OCR text in your image, consider increasing the Text Mask Buffer.

<details>
  <summary>Technical Notes for Developers.</summary>
  <div><br>
    
**Default imports:**
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
    
  </div>
</details>
