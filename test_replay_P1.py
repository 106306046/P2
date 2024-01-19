import matplotlib
import matplotlib.pyplot as plt

import time
import random
from pylsl import StreamInfo, StreamOutlet
from PIL import Image

#for SSVEP
import pygame
from math import sin, pi
import numpy as np

#for sound
import platform
if platform.system() == 'Windows':
    import winsound
else:
    import subprocess


def play_start_sound(cue):
    if platform.system() == 'Windows':
        #dict = ['LH', 'RH', 'F', 'BH']
        sound = "../P1_20231109/sound/cue_" + cue + ".wav"
        winsound.PlaySound(sound,winsound.SND_FILENAME)
        sound = "../P1_20231109/sound/button04a.wav"
        winsound.PlaySound(sound,winsound.SND_FILENAME)
    else:
        sound = "../P1_20231109/sound/cue_" + cue + ".wav"
        subprocess.run(["afplay", sound])
        sound = "../P1_20231109/sound/button04a.wav"
        subprocess.run(["afplay", sound])

def play_finish_sound():
    if platform.system() == 'Windows':
        sound = "../P1_20231109/sound/button02a.wav"
        winsound.PlaySound(sound,winsound.SND_FILENAME)
    else:
        sound = "../P1_20231109/sound/button02a.wav"
        subprocess.run(["afplay", sound])

#initial parameter
random.seed(18796441854)
trials_per_class = 48
initial_time = 2

perform_time = 3.2
eog_perform_time = 30
wait_time = 2
pause_every = 40
pause_duration = 20
fontsize = 10
MI_labels = ['LH', 'RH', 'F', 'BH']
trial_labels = ['LH', 'RH', 'F', 'BH', 'SSVEP']
#markers = {'LH':'left hand',  :'right hand', 'F':'foot', 'T':'tongue'}

#隨機SSVEP頻率 (0表示不顯示有12組,其他四個頻率9組)
SSVEP_list = np.array([0,6,4.3,7.6,10])
SSVEP_list = np.repeat(SSVEP_list,6)
SSVEP_list = np.append(SSVEP_list,[0,0,0,0,0,0,0,0,0,
                                   6,6,6,6,6,6,6,6,6])
random.shuffle(SSVEP_list)
SSVEP_IDX = 0
if len(SSVEP_list) !=trials_per_class:
    print("SSVEP length is not match!")

#產生實驗順序
labels_arr = []
for j in range(int(trials_per_class*len(trial_labels)/pause_every)):
    run_arr = []
    for i in range(int(pause_every/len(trial_labels))):
        for label in trial_labels:
            run_arr.append(label)
    random.shuffle(run_arr)
    print(run_arr)
    labels_arr.append(run_arr)
def flatten(l):
    return [item for sublist in l for item in sublist]
labels_arr = flatten(labels_arr)


# load image
LH_img = Image.open("../P1_20231109/icon/fist_L.png")
RH_img = Image.open("../P1_20231109/icon/fist_R.png")
T_img  = Image.open("../P1_20231109/icon/fist_Both.png")
F_img  = Image.open("../P1_20231109/icon/foot.png")
matplotlib.rcParams.update({'font.size': fontsize})




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

# fullscreen
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()

plt.ion()
plt.draw()
plt.show()

#manager = plt.get_current_fig_manager()
#manager.full_screen_toggle()
hFigure.canvas.draw()
hFigure.canvas.flush_events()

print("Press [Enter] to begin.")
x = input()

labels_arr = labels_arr[160:]
try:
 # Calibration trial
    for trial, cur_label in enumerate(labels_arr):
        if not plt.fignum_exists(hFigure.number): 
            break

        ax.set_title(f"Trial: {trial+1}",loc = "left", fontsize = 10, fontname = 'monospace')
        print(f"Trial {trial+1} | {cur_label}")

        # initial state
        
        t.set_text("+")
        hFigure.canvas.draw()
        hFigure.canvas.flush_events()
        start_time = time.time()
        play_start_sound(cur_label)
        #time.sleep(initial_time)
        end_time = time.time()     
        print(f"initial state 實際執行時間為: {end_time - start_time} 秒")

        # imaging state
        if cur_label == 'SSVEP':
            t.set_text(str(SSVEP_list[SSVEP_IDX]))
            hFigure.canvas.draw()
            hFigure.canvas.flush_events()
            start_time = time.time()
            print("freq is " + str(SSVEP_list[SSVEP_IDX]) + 'Hz')
            time.sleep(perform_time)
            SSVEP_IDX += 1
        else:
            t.set_text("")
            for label in MI_labels:
                imgs[label].set_alpha(1 if label == cur_label else 0)
            hFigure.canvas.draw()
            hFigure.canvas.flush_events()
            start_time = time.time()
            time.sleep(perform_time)
            for label in MI_labels:
                imgs[label].set_alpha(0)
        end_time = time.time()
        print(f"imaging state 實際執行時間為: {end_time - start_time} 秒")

        # break state
        if trial+1 == len(labels_arr):
            for s in range(10,0,-1):
                t.set_text(f'Calibration End\n\n{s}')
                hFigure.canvas.draw()
                hFigure.canvas.flush_events()
                if s == 0 :
                    play_finish_sound()
                time.sleep(1)
            t.set_text('')
            hFigure.canvas.draw()
            hFigure.canvas.flush_events() 
        elif (trial+1) % pause_every == 0:
            for s in range(0,pause_duration+1):
                t.set_text(f'Rest {pause_duration} sec\n\nRemain {pause_duration-s} sec')
                hFigure.canvas.draw()
                hFigure.canvas.flush_events()
                if s == 0 :
                    play_finish_sound()
                time.sleep(1)
            t.set_text('')
            hFigure.canvas.draw()
            hFigure.canvas.flush_events() 
        else:
            start_time = time.time()
            t.set_text(f'+')
            hFigure.canvas.draw()
            hFigure.canvas.flush_events()
            play_finish_sound()
            #time.sleep(wait_time) 
            end_time = time.time()     
            print(f"break state 實際執行時間為: {end_time - start_time} 秒")
        
except Exception as e:
    print(e)
