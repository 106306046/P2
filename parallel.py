from P2_draw_client import main as whiteboard
from call_api import main as api
#from P2_draw_sever import main as sever
from pathlib import Path
import threading

PICKLE_PATH = Path("./sample_data/tsun_1228.pickle")
KERAS_PATH = Path("./sample_data/tsun_1228.h5")
SAMPLELSL_PATH = Path("./sample_data/tsun_1228_test_mock_lsl.fif")

t1 = threading.Thread(target = api)
t2 = threading.Thread(target = whiteboard)
#t3 = threading.Thread(target = sever, args=(PICKLE_PATH, KERAS_PATH, SAMPLELSL_PATH))

t1.start()
t2.start()
#t3.start()