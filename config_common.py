#
# Common configuration code for reepo
#
# 1) Configures Python Path
# 2) Configures key configuration variables:
#     - BASE_DIR                    (location of root directory for this repo)
#     - physionet_db_name           (name of dataset on physionet.org)
#     - local_recordings_dir_full   (samples)
#     - media_recordings_dir_full   (full recording set)



import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# add source directories to path
sys.path.append(os.path.join(BASE_DIR, 'src'))

# name of physionet dataset
physionet_db_name = 'ctu-uhb-ctgdb'

# location of local copy of samples
local_recordings_dir = os.path.join('local_data', 'sample_physionet_ctb_uhb_recordings')
local_recordings_dir_full =os.path.join(BASE_DIR, local_recordings_dir)

# loocation of full local copy of dataset (on removable media due to size)
media_recordings_dir_full ='/Volumes/Recordings/physionet/ctu-uhb-ctgdb'