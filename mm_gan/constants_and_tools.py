import os
from pathlib import Path

TXT_NAMES = ['group0.txt', 'group1.txt', 'group2.txt', 'group3.txt', 'group4.txt']
TXT_OUT_NAMES = ['group0_standardized.txt', 'group1_standardized.txt', 'group2_standardized.txt',
                 'group3_standardized.txt', 'group4_standardized.txt']
CHANNELS = ['t1', 't2', 't1ce', 'flair']
NII_SHAPE = (155, 240, 240)
SLICE_RANGE = (12, 148)


def make_dir(absolute_dir: str):
    if len(absolute_dir.split('.')) > 1:  # if this is a file, use its parent; don't use file without suffix
        absolute_dir = os.path.abspath(os.path.dirname(absolute_dir) + os.path.sep + ".")
    if Path(absolute_dir).is_dir():
        return
    parent_dir = os.path.abspath(os.path.dirname(absolute_dir) + os.path.sep + ".")
    if Path(parent_dir).is_dir():
        os.mkdir(absolute_dir)
    else:
        make_dir(parent_dir)
