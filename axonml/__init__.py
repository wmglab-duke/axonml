import glob
import os

this = os.path.dirname(__file__)
all_trained = glob.glob(this + "/trained/*")

trained = {os.path.split(t)[1]: t for t in all_trained}
