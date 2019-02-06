#
# Config Values
#

levels_up = 2 # position of base directory above


#
# Set Python Path to include Base Dir
#

import os, sys
# Extract base directory using position relative to current directory
BASE_DIR = os.getcwd()
for i in range(levels_up):
    BASE_DIR = BASE_DIR[:BASE_DIR.rfind(os.sep)]

sys.path.append(BASE_DIR)