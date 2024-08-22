"""
This file contains the paths to file resources, used in the GUI or by the package.
"""

import os

root = os.path.dirname(os.path.abspath(__file__))

ICONS = {
    # "normalisation": os.path.normpath(os.path.join(root, "gui/icons/flaticon/normalization_b&w.png")),
    # "normalisation": os.path.normpath(os.path.join(root, "gui/icons/normalisation.png")),
    "normalisation_light": os.path.normpath(
        os.path.join(root, "gui/icons/normalisation2_light.png")
    ),
    "normalisation_dark": os.path.normpath(
        os.path.join(root, "gui/icons/normalisation2_dark.png")
    ),
}
