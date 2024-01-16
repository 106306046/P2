import mne
import pyxdf
from mnelab.io import read_raw

fname = "/content/drive/MyDrive/BCI/Recordings/tsun_1228.xdf"

#決定要用哪一組steam
#1 means unfilter, 2 uses filter by OPENBCI GUI, 3 is aux
used_eeg = 'obci_eeg2'

streams, header = pyxdf.load_xdf(fname)
for i in streams:
    if i['info']['name'][0] == used_eeg:
        eeg_id = i['info']['stream_id']
        break

raw_ori = read_raw(fname, stream_ids=[eeg_id], fs_new=125)
raw_ori.set_channel_types({used_eeg + '_0':'eeg', used_eeg + '_1':'eeg',
                           used_eeg + '_2':'eeg', used_eeg + '_3':'eeg',
                           used_eeg + '_4':'eeg', used_eeg + '_5':'eeg',
                           used_eeg + '_6':'eeg', used_eeg + '_7':'eeg',
                           used_eeg + '_8':'eeg', used_eeg + '_9':'eeg',
                           used_eeg + '_10':'eeg',used_eeg + '_11':'eeg',
                           used_eeg + '_12':'eeg',used_eeg + '_13':'eeg',
                           used_eeg + '_14':'eeg',used_eeg + '_15':'eeg'})
raw_ori.rename_channels({used_eeg + '_0':'0',  used_eeg + '_1':'1',
                         used_eeg + '_2':'2', used_eeg + '_3':'3',
                         used_eeg + '_4':'4',  used_eeg + '_5':'5',
                         used_eeg + '_6':'6',  used_eeg + '_7':'7',
                         used_eeg + '_8':'8',  used_eeg + '_9':'9',
                         used_eeg + '_10':'10', used_eeg + '_11':'11',
                         used_eeg + '_12':'12', used_eeg + '_13':'13',
                         used_eeg + '_14':'14',used_eeg + '_15':'15'})

raw = raw_ori.copy()
events,all_events_id=mne.events_from_annotations(raw,event_id='auto')
for event in events:
  if event[2] ==all_events_id['Warm-up-begin']:
    crop_start = event[0]
    break
c_raw = raw.copy().crop(tmin=crop_start/125)
c_raw.save('tsun_1228_raw_crop.fif',overwrite=True)