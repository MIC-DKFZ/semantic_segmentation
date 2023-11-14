from os.path import join, dirname
import sys

this_dir = dirname(__file__)

src_dir = join(this_dir, "..")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
