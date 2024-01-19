import time
from datetime import datetime
from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt

import mne
from mne_lsl.stream import StreamLSL as Stream
from mne_lsl.player import PlayerLSL as Player
from P2_model import MI_pred
from P2_model import fbcca_realtime

import tensorflow as tf
import pickle
from PIL import Image


PICKLE_PATH = Path("./sample_data/bo_1231_160trials.pickle")
KERAS_PATH = Path("./sample_data/bo_1231_160trials.h5")
SAMPLELSL_PATH = Path("./sample_data/bobo_1231_final80trials_raw.fif")
#SAMPLELSL_PATH = Path("./sample_data/bobo_1231_test_mock_lsl_raw.fif")

LOG_PATH = Path("./log.txt")

# this is to end this program
end = 0

#計算時間相關
window_size = 2
pred_interval = 0.8

#SSVEP_parameter
list_freqs = [6,4.3,7.6,10]
num_harms = 9
num_fbs = 7

#MI_parameter
with open(PICKLE_PATH, 'rb') as f:
    binary_class_dict = pickle.load(f)
    band_dict = pickle.load(f)
    CSPs = pickle.load(f)
model = tf.keras.saving.load_model(KERAS_PATH)

#清除log文件歷史資料
with open(LOG_PATH,'a+') as f:
    f.truncate(0)
    f.write('0 0 idle idle\n')

player = Player(SAMPLELSL_PATH)
player.start()
stream = Stream(bufsize=2)
raw = mne.io.read_raw(SAMPLELSL_PATH)
player_length = len(raw)/125
mode = 'rp'
print('starting replay...')

LH_img = Image.open("../P1_20231109/icon/fist_L.png")
RH_img = Image.open("../P1_20231109/icon/fist_R.png")
T_img  = Image.open("../P1_20231109/icon/fist_Both.png")
F_img  = Image.open("../P1_20231109/icon/foot.png")
hFigure, ax = plt.subplots()
plt.axis('off')
#ax.set_yticklabels([''])
#ax.set_xticklabels([''])
t = plt.text(5, 5, '', ha = 'center', va = 'center', fontsize=20)
plt.xlim(xmin=-0.5, xmax=10.5)
plt.ylim(ymin=-0.5, ymax=10.5)
imgs = {}
imgs["LH"] = ax.imshow(LH_img, alpha = 0.1, extent=(0, 3, 3.75, 6.25))
imgs["RH"] = ax.imshow(RH_img, alpha = 0.1, extent=(7, 10, 3.75, 6.25))
imgs["BH"] = ax.imshow(T_img, alpha = 0.1, extent=(3.75, 6.25, 7, 10))
imgs["F"] = ax.imshow(F_img, alpha = 0.1, extent=(3.75, 6.25, 0, 3))


plt.ion()
plt.draw()
plt.show()

hFigure.canvas.draw()
hFigure.canvas.flush_events()

print("Press [Enter] to begin.")
x = input()

stream.connect(acquisition_delay=0.001)
stream.rename_channels({'0':'O1', '1':'O2', '2':'PO3', '3':'PO4',
                        '4':'P3', '5':'P4', '6':'C3',  '7':'C4',
                        '8':'F3', '9':'F4', '10':'T3', '11':'T4',
                        '12':'Cz','13':'Pz','14':'CP1','15':'CP2'})
stream.set_montage("standard_1020")
raw_info = stream.info
sfreq = 125

#開始predict
start_time = time.time()
if mode == 'rp':
    start_loop = time.time()
while (end!=True):

    #至少每X秒計算一次
    cur_time = time.time()
    if cur_time - start_time < pred_interval:
        continue
    start_time = time.time()

    #記錄開算時間
    compute_start = time.time()

    #read realtime data
    data = stream.get_data(window_size)[0]
    rt_epoch = mne.EpochsArray([data], raw_info)
    rt_epoch.reorder_channels(['O1','O2','PO3','PO4','P3','Pz','P4','CP1',
                               'CP2','T3','C3','Cz','C4','T4','F3','F4'])
    rt_epoch.apply_function(lambda x: x * 1e-6/10)


    # 順序計算
    mi_pred = MI_pred(rt_epoch,model,binary_class_dict,band_dict,CSPs,sfreq)
    ssvep_pred = fbcca_realtime(rt_epoch.get_data(copy = True)[0], list_freqs, sfreq,num_harms,num_fbs)

    #記錄預測結果
    mi_pred_dict = {0:'BH', 1:'F', 2:'LH', 3:'RH', 4:'idle'}
    mi_output = mi_pred_dict[mi_pred]
    ssvep_pred_dict = {0:'6', 1:'4.3', 2:'7.6', 3:'10', 4:'idle'}
    ssvep_output = ssvep_pred_dict[ssvep_pred]
    #print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #print(f"MI_pred :{mi_pred}, output: {mi_output}" )
    #print(f"SSVEP_pred :{ssvep_pred}, output: {ssvep_output}" )
    #print("++++++++++++++++++++++++++++++++++++++++++++++++++++")

    MI_labels = ['LH', 'RH', 'F', 'BH']
    t.set_text(ssvep_output)
    for label in MI_labels:
        imgs[label].set_alpha(1 if label == mi_output else 0)
    hFigure.canvas.draw()
    hFigure.canvas.flush_events()
    for label in MI_labels:
        imgs[label].set_alpha(0)


    #輸出到log檔
    current_time = datetime.now()
    with open(LOG_PATH, 'a') as f:
        f.write(str(current_time) + ' ' + str(mi_output) + ' ' + str(ssvep_output) + '\n')

    #計算預測運算時間
    compute_fin = time.time()
    compute_time = compute_fin - compute_start
    print(f"計算時間為：{compute_time}秒")


    if mode == 'rp':
        if compute_fin-start_loop >= player_length:
            end = 1
stream.disconnect()
if mode == 'rp':
    player.stop()

