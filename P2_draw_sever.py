import time
from datetime import datetime
from pathlib import Path

import mne
from mne_lsl.stream import StreamLSL as Stream
from mne_lsl.player import PlayerLSL as Player
from P2_model import MI_pred
from P2_model import fbcca_realtime

import tensorflow as tf
import pickle

PICKLE_PATH = Path("./sample_data/tsun_1228.pickle")
KERAS_PATH = Path("./sample_data/tsun_1228.keras")
SAMPLELSL_PATH = Path("./sample_data/tsun_1228_test_mock_lsl_raw.fif")
LOG_PATH = Path("./log.txt")

def main():
    # this is to end this program
    end = 0

    #計算時間相關
    window_size = 2
    pred_interval = 1

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

    #Replay or Stream
    while 1:
        print('please input mode:\n 1:stream; 2:replay')
        X = input()
        if X == "1":
            host_id = 'openbcigui'
            used_eeg = "obci_eeg2"
            stream = Stream(name=used_eeg, stype='EEG', source_id=host_id, bufsize=5)  # 5 seconds of buffer
            mode = 'st'
            print('starting streaming...')
            break
        elif X == "2":
            player = Player(SAMPLELSL_PATH)
            player.start()
            stream = Stream(bufsize=3)
            raw = mne.io.read_raw(SAMPLELSL_PATH)
            player_length = len(raw)/125
            mode = 'rp'
            print('starting replay...')
            break




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

        # 平行計算(報錯)
        '''   
        q = mp.Queue()
        p1 = mp.Process(target=MI_pred, args=(rt_epoch,model,binary_class_dict,band_dict,CSPs,sfreq))
        p2 = mp.Process(target=fbcca_realtime, args=(rt_epoch.get_data(copy = True)[0], list_freqs, sfreq,num_harms,num_fbs))
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        mi_pred = q.get()
        ssvep_pred = q.get()
        '''
        # 順序計算
        mi_pred = MI_pred(rt_epoch,model,binary_class_dict,band_dict,CSPs,sfreq)
        ssvep_pred = fbcca_realtime(rt_epoch.get_data(copy = True)[0], list_freqs, sfreq,num_harms,num_fbs)

        #記錄預測結果
        mi_pred_dict = {0:'Up', 1:'Down', 2:'Left', 3:'Right', 4:'idle'}
        mi_output = mi_pred_dict[mi_pred]
        ssvep_pred_dict = {0:'E', 1:'A', 2:'B', 3:'D', 4:'idle'}
        ssvep_output = ssvep_pred_dict[ssvep_pred]
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"MI_pred :{mi_pred}, output: {mi_output}" )
        print(f"SSVEP_pred :{ssvep_pred}, output: {ssvep_output}" )
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++")


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

if __name__ == '__main__':
    main()