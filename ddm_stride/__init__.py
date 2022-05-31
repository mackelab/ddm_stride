import os
import pathlib

os.environ["DDM_STRIDE_DIR"] = str(pathlib.Path(__file__).parent.parent.absolute())
