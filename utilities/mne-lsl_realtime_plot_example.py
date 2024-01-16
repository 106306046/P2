import time
from matplotlib import pyplot as plt
from mne_lsl.stream import StreamLSL as Stream
import mne

host_id = 'openbcigui'
used_eeg = "obci_eeg2"
stream = Stream(name=used_eeg,stype = 'EEG',source_id = host_id,  bufsize=5)  # 5 seconds of buffer
stream.connect(acquisition_delay=0.001)
stream.rename_channels({'0':'O1', '1':'O2', '2':'PO3', '3':'PO4',
                        '4':'P3', '5':'P4', '6':'C3',  '7':'C4',
                        '8':'F3', '9':'F4', '10':'T3', '11':'T4',
                        '12':'Cz','13':'Pz','14':'CP1','15':'CP2'})
stream.set_montage("standard_1020")
info = stream.info
sfreq = info['sfreq']

plot_time = 5

while 1:

    rt_raw = mne.io.RawArray(stream.get_data(plot_time)[0], info)
    rt_raw.reorder_channels(['O1', 'O2', 'PO3', 'PO4', 'P3', 'Pz', 'P4', 'CP1',
                             'CP2', 'T3', 'C3', 'Cz', 'C4', 'T4', 'F3', 'F4'])
    rt_raw.apply_function(lambda x: x * 1e-6/10)

    f, ax = plt.subplots(16, 1, sharex=True, constrained_layout=True)
    for k, data_channel in enumerate(rt_raw.get_data()):
        ax[k].cla()
        ax[k].plot(range(0, sfreq*plot_time), data_channel)
    plt.pause(3)
    plt.draw()