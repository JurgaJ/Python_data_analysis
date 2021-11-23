# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light,md:myst
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Biological data analysis
#
# 2020-10-05
#
# Send solutions (ipynb and pdf or html) till 2020-10-10 23:55 to
# avoicikas@gmail.com

# Fill in your name:

# Evaluation:
#
# - Comments
# - Applied methods
# - Figures
# - Results

# ---
# >> **TASK**
# >>
# >> - Import and fix the speech data.
# >>
# >> - How different methods dealing with the missing data
# >> (interpolation etc.) affect the time-frequency plots of the data.
# >>
# >> Data file exam/A5/speech.csv
# >>
#
# ---

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from scipy import signal

df=pd.read_csv('./speech.csv')
df.columns
# -

sf=7418
df.head()

df=df.drop(labels=' V', axis=1).drop(index=0,axis=1)
df.head()

df=df.iloc[:,0].str.strip('#').str.strip('V').astype(float)
df.head()

t=np.linspace(0,df.shape[0]/sf,df.shape[0])
plt.plot(t,df)
plt.xlabel('Time (s)');

freq = np.linspace(1, sf/2, 50)
w = 20
widths = w * (sf) / (2 * freq * np.pi)
cwtm = signal.cwt(df, signal.morlet2, widths, w=w)
plt.pcolormesh(t, freq, np.abs(cwtm), cmap="viridis");


df[df==-100] = np.nan

# Interpolate

dff = df.copy().interpolate(method='pchip')
plt.plot(t,dff)
plt.xlabel('Time (s)');

freq = np.linspace(1, sf/2, 50)
w = 20
widths = w * (sf) / (2 * freq * np.pi)
cwtm = signal.cwt(dff, signal.morlet2, widths, w=w)
plt.pcolormesh(t, freq, np.abs(cwtm), cmap="viridis")
plt.show()

# Interpolate with nearest neighbours

dff = df.copy().interpolate(method='nearest')
plt.plot(t,dff)
plt.xlabel('Time (s)');

freq = np.linspace(1, sf/2, 50)
w = 20
widths = w * (sf) / (2 * freq * np.pi)
cwtm = signal.cwt(dff, signal.morlet2, widths, w=w)
plt.pcolormesh(t, freq, np.abs(cwtm), cmap="viridis")
plt.show()

# Delete

dff = df.copy().dropna()
t=np.linspace(0,dff.shape[0]/sf,dff.shape[0])
plt.plot(t,dff)
plt.xlabel('Time (s)');

freq = np.linspace(1, sf/2, 50)
w = 20
widths = w * (sf) / (2 * freq * np.pi)
cwtm = signal.cwt(dff, signal.morlet2, widths, w=w)
plt.pcolormesh(t, freq, np.abs(cwtm), cmap="viridis");

# Fill with mean

dff = df.copy().fillna(df.mean())
t=np.linspace(0,dff.shape[0]/sf,dff.shape[0])
plt.plot(t,dff)
plt.xlabel('Time (s)');

freq = np.linspace(1, sf/2, 50)
w = 20
widths = w * (sf) / (2 * freq * np.pi)
cwtm = signal.cwt(dff, signal.morlet2, widths, w=w)
plt.pcolormesh(t, freq, np.abs(cwtm), cmap="viridis");

# Fill with samples from data

dff = df.copy().apply(lambda x: np.random.choice(df.copy().dropna().values) if np.isnan(x) else x)
t=np.linspace(0,dff.shape[0]/sf,dff.shape[0])
plt.plot(t,dff)
plt.xlabel('Time (s)');

freq = np.linspace(1, sf/2, 50)
w = 20
widths = w * (sf) / (2 * freq * np.pi)
cwtm = signal.cwt(dff, signal.morlet2, widths, w=w)
plt.pcolormesh(t, freq, np.abs(cwtm), cmap="viridis");

# ---
# >> **TASK**
# >>
# >> EEG power bands analysis.
# >>
# >> ![powerbands](./powerBandsEEG.jpeg)
# >>
# >> - Calculate average power in these intervals.
# >> - Compare eyes open (EO) and eyes closed (EC) conditions.
# >>
# >> exam/A5/EO.csv and EC.csv or set/fdt matlab files
# ---

# +
import mne
import pathlib

def import_eeg(ifile):
    data = mne.io.read_raw_eeglab(ifile, eog='auto')
    data.load_data()
    try:
        data.set_channel_types({'EKG': 'ecg'})
        ecg=1
    except Exception:
        print('no ecg')
        ecg=0
    data.set_montage('standard_1020')
    data.set_eeg_reference(ref_channels='average')
    return data, ecg


# -

data_dir = './'
files = list(pathlib.Path(data_dir).glob('*.set'))
files

dataEC, ecgon = import_eeg(str(files[0]))
dataEO, ecgon = import_eeg(str(files[1]))
dataEC.plot_psd(fmin=2,fmax=60);

dataEO.plot_psd(fmin=2,fmax=60);


def eeg_power_band(data):
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "beta": [15.5, 30],
                  "gamma": [30, 50],
                  }
    psds, freqs = mne.time_frequency.psd_welch(data, picks='eeg', fmin=0.5, fmax=50.)
    psds /= np.sum(psds, axis=-1, keepdims=True)
    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))
    return np.concatenate(X, axis=1)


bandsEO=pd.DataFrame(eeg_power_band(dataEO), columns=['delta','theta','alpha','beta','gamma'],index=dataEC.ch_names)
bandsEC=pd.DataFrame(eeg_power_band(dataEC), columns=['delta','theta','alpha','beta','gamma'],index=dataEC.ch_names)
bandsDiff=bandsEC-bandsEO
bandsEC.shape

bandsEC.head()

bandsDiff.plot(kind='box');

bandsDiff.plot(kind='bar',figsize=(15,15));

# ---
# >> **TASK**
# >>
# >> Earthquakes and explosions present similar time-domain signals.
# >> It is important to differentiate between these events.
# >>
# >> Data file exam/A5/ExplosionOrEarthquake.csv contains 16 recordings.
# >> First 8 columns are seismic recordings of earthquake and the last 8 of
# >> explosions.
# >>
# >>
# >> Find:
# >>
# >> - The start time of the earthquake.
# >> - Frequency and power of the earthquake.
# >> - Duration of the earthquake.
# >> - Compare earthquakes with explosions.
# >>
#
# ---

data=pd.read_csv('ExplosionOrEarthquake.csv', names=[str(x) for x in range(1,17)])
data.head()

data.shape

sf=2048
t=np.linspace(0,data.shape[0]/sf,data.shape[0])
fig ,ax = plt.subplots(8, 2,sharex=True,sharey=True)
ax=ax.ravel(order='F')
for ii,iax in enumerate(ax):
  ax[ii].plot(t,data.iloc[:,ii])
fig.suptitle('Earthquakes vs Explosions');

data.plot(kind='box');

freq = np.linspace(1, sf/2, 50)
w = 3
widths = w * (sf) / (2 * freq * np.pi)
cwtm=[]
for element in range(16):
  cwtm.append(np.abs(signal.cwt(data.iloc[:,element], signal.morlet2, widths, w=w)))
cwtm = np.array(cwtm)

sf=2048
t=np.linspace(0,data.shape[0]/sf,data.shape[0])
fig ,ax = plt.subplots(8, 2,sharex=True,sharey=True,figsize=(15,15))
ax=ax.ravel(order='F')
for ii,iax in enumerate(ax):
  ax[ii].pcolormesh(t, freq, cwtm[ii,:,:], cmap="viridis")
fig.suptitle('Earthquakes vs Explosions TF');

sf=2048
t=np.linspace(0,data.shape[0]/sf,data.shape[0])
fig ,ax = plt.subplots(8, 2,sharex=True,sharey=True,figsize=(10,10))
ax=ax.ravel(order='F')
for ii,iax in enumerate(ax):
  ax[ii].plot(t,  np.mean(cwtm[ii,:,:],axis=0))
fig.suptitle('Earthquakes vs Explosions time flow');

sf=2048
t=np.linspace(0,data.shape[0]/sf,data.shape[0])
fig ,ax = plt.subplots(8, 2,sharex=True,sharey=True,figsize=(10,10))
ax=ax.ravel(order='F')
for ii,iax in enumerate(ax):
  ax[ii].plot(freq,  np.mean(cwtm[ii,:,:],axis=1))
fig.suptitle('Earthquakes vs Explosions frequencies');

# ---
# >> **TASK**
# >>
# >> Recorded neural response to rapidly changing stimulus (~200 trials).
# >>
# >> exam/A5/itpc.csv
# >>
# >> Find:
# >>
# >> - Stimulus parameters (duration, frequency).
# >> - Time and frequency range of maximum response.
# >> - Time and frequency when maximum syncrony with the stimulus occurs.
#
# ---

data=pd.read_csv('itpc.csv')
data.head()

df=data.pivot_table(values='FCz',index="time",columns='epoch')
df.head()

sf=1000
df.index

erp=df.mean(axis=1)
erp.plot();

freq = np.linspace(1, sf/2, 500)
w = 20
widths = w * (sf) / (2 * freq * np.pi)
cwtm=np.abs(signal.cwt(erp, signal.morlet2, widths, w=w))**2
cwtm_base = np.mean(cwtm[:,0:900],axis=1)
plt.pcolormesh(erp.index, freq, cwtm, cmap="viridis")
plt.ylim(1,100);

freq = np.linspace(1, sf/2, 500)
w = 20
widths = w * (sf) / (2 * freq * np.pi)
cwtm=np.abs(signal.cwt(erp, signal.morlet2, widths, w=w))**2
cwtm_base = np.mean(cwtm[:,0:900],axis=1)
plt.pcolormesh(erp.index, freq, cwtm/np.expand_dims(cwtm_base,axis=1), cmap="viridis")
plt.ylim(1,100);

# +
sf=1000
freq = np.linspace(1, sf/2, 100)
w = 20
widths = w * (sf) / (2 * freq * np.pi)
cwtm=[]
for element,_ in enumerate(df):
  cwtm.append(signal.cwt(df.iloc[:,element], signal.morlet2, widths, w=w))

A = np.abs(np.mean(np.exp(1j*np.angle(cwtm)), axis=0))
plt.pcolormesh(
    erp.index.values,
    freq,
    A,
    cmap="viridis",
);
plt.ylim([0, 100])
plt.colorbar();
