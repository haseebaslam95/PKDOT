import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pickle
import os
import tqdm
import librosa.display
import librosa
import torch
import glob
import biosppy.signals.emg as emg
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
from scipy.signal import cwt, morlet, spectrogram
from torchvision import transforms

root_path_0 = '/home/livia/work/Biovid/PartB/biovid_classes/physio/0'
root_path_1 = '/home/livia/work/Biovid/PartB/biovid_classes/physio/4'

#list of all files in the directory
file_list_0 = os.listdir(root_path_0)
file_list_1 = os.listdir(root_path_1)

#randomly selec5 5 files from each directory
random_files_0 = np.random.choice(file_list_0, 5)
random_files_1 = np.random.choice(file_list_1, 5)


# path_0= '/home/livia/work/Biovid/PartB/biovid_classes/physio/0/071614_m_20-BL1-082_bio.csv'
# path_1= '/home/livia/work/Biovid/PartB/biovid_classes/physio/4/071614_m_20-PA4-039_bio.csv'


#create a function for the following code
def plot_physio(path):
    #code to read csv file  
    emg_path = path
    #read  csv file with header
    emg_df = pd.read_csv(emg_path,sep="\t", header=0,index_col=False)

    #plot the emg signal 
    plt.plot(emg_df['emg_corrugator'])
    plt.show()

    print(emg_df['emg_corrugator'])



def biosppy_check(path):
    emg_path = path
    #read  csv file with header
    emg_df = pd.read_csv(emg_path,sep="\t", header=0,index_col=False)

    # Sample EMG data (replace with your actual data)
    emg_data = emg_df['emg_corrugator']

    # Process the EMG signal using biosppy
    out = emg.emg(emg_data, sampling_rate=512,show=True)
    sampling_rate = len(emg_df) / 5.5
    print(f"Sampling rate: {sampling_rate}")

    # Extracted features
    ts, filtered, onsets  = out

    # Plot the processed EMG signal
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(ts, emg_data, label='Raw EMG')
    plt.title('Raw EMG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.plot(ts, filtered, label='Rectified EMG', color='green')
    plt.title('Rectified EMG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 3)
    plt.plot(ts, filtered, label='Processed EMG', color='red')
    plt.title('Processed EMG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

    # Extracted features
    print(f"Onsets: {onsets}")
    # print(f"Offsets: {offsets}")




def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def filter_emg(physio_df):
    fs = 1000  # Sampling frequency (Hz)
    cutoff = 40  # Cutoff frequency for low-pass filter (Hz)
    # emg_path = path
    # emg_df = pd.read_csv(emg_path,sep="\t", header=0,index_col=False)

    #read samples only from 2 seconds to 5 seconds
    ts=physio_df['time']
    #convert ts from microseconds to milliseconds
    ts=ts/1000

    #only read emg values where 2<=ts<=5
    physio_df=physio_df[(ts>=2000) & (ts<=5000)]
    emg_data=physio_df['emg_corrugator']
    # filtered_emg = butter_lowpass_filter(emg_data, cutoff, fs)
    filtered_emg = emg_data
    return filtered_emg



def get_statistical_features(path):
    filtered_emg=filter_emg(path)
    mean_value = np.mean(filtered_emg)
    variance = np.var(filtered_emg)
    rms = np.sqrt(np.mean(filtered_emg**2))
    skewness = skew(filtered_emg)
    kurt = kurtosis(filtered_emg)
    print(f"Mean: {mean_value}")
    print(f"Variance: {variance}")
    print(f"RMS: {rms}")
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurt}")



def get_cwt_scalograms(emg_data):
    # Generate some example EMG data (replace this with your actual data)

    # Define parameters for the CWT
    widths = np.arange(1, 512)  # Widths for the CWT (corresponding to different frequencies)
    emg_data = emg_data
    # Compute the CWT
    coefficients = cwt(emg_data, morlet, widths)
    n_channels = 1  # Since it's a grayscale image
    num_time_steps, num_frequencies = coefficients.shape
    cwt_2d = coefficients.reshape((num_time_steps, num_frequencies, n_channels))
    cwt_2d=cwt_2d.squeeze(2)
    # cwt_2d = torch.from_numpy(cwt_2d)
    cwt_2d_reshaped = cwt_2d.reshape((511,1536))




    # # Plot the CWT coefficients
    # plt.figure(figsize=(10, 6))
    # plt.imshow(np.abs(coefficients), aspect='auto', extent=[0, len(emg_data), widths[-1], widths[0]], cmap='jet')
    # plt.colorbar(label='Magnitude')
    # plt.title('Continuous Wavelet Transform (CWT) of EMG Signal')
    # plt.ylabel('Scale')
    # plt.xlabel('Time')
    # plt.show()
    return cwt_2d_reshaped

def get_spectrograms(emg_data):

    emg_data = emg_data

    # Define parameters for spectrogram calculation
    fs = 1000  # Sampling frequency in Hz (adjust according to your data)
    nperseg = 16  # Number of data points per segment for spectrogram calculation
    noverlap = 4  # Overlap between segments
    nfft = 128  # Number of data points used in each block for the FFT

    # Compute the spectrogram
    n_channels = 1  # Since it's a grayscale image
    frequencies, times, Sxx = spectrogram(emg_data, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    n_time_steps, n_freq_bins = Sxx.shape
    spectrogram_2d = Sxx.reshape((n_channels, n_time_steps, n_freq_bins))

    # Visualize the spectrogram
    # plt.figure(figsize=(10, 6))
    # plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx))  # Convert to dB scale
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.colorbar(label='Intensity [dB]')
    # plt.title('EMG Spectrogram')
    # plt.show()
    return spectrogram_2d


def open_physio_df(path):
    df_path = path
    #read  csv file with header
    physio_df = pd.read_csv(df_path,sep="\t", header=0,index_col=False)
    return physio_df



# *******************************FUNCTION CALLS***********************************

#open files and create dataframes random file 0 and random file 1

# specs_plot=[]
# for i in range(len(random_files_0)):
#     path_0= os.path.join(root_path_0, random_files_0[i])
#     path_1= os.path.join(root_path_1, random_files_1[i])
#     physio_df_0= open_physio_df(path_0)
#     physio_df_1= open_physio_df(path_1)
#     filtered_emg_0=filter_emg(physio_df_0)
#     filtered_emg_1=filter_emg(physio_df_1)
#     specs_plot.append(get_spectrograms(filtered_emg_0))
#     specs_plot.append(get_spectrograms(filtered_emg_1))

#plot spectrograms
# for i in range(len(specs_plot)):
#     plt.imshow(specs_plot[i])
#     plt.show()



# path_0= os.path.join(root_path_0, random_files_0[0])
# path_1= os.path.join(root_path_1, random_files_1[0])

path_0 = '/home/livia/work/Biovid/PartB/biovid_classes/physio/0/071614_m_20-BL1-082_bio.csv'
path_1 = '/home/livia/work/Biovid/PartB/biovid_classes/physio/4/071614_m_20-PA4-039_bio.csv'

physio_df_0= open_physio_df(path_0)
physio_df_1= open_physio_df(path_1)

filtered_emg_0=filter_emg(physio_df_0)
get_spectrograms(filtered_emg_0)
# # cwt=get_cwt_scalograms(filtered_emg_0)

  
filtered_emg_1=filter_emg(physio_df_1)
get_spectrograms(filtered_emg_1)
# get_cwt_scalograms(filtered_emg_1)




# biosppy_check(path_0)
# plot_physio(path_0)
# get_statistical_features(path)



# def get_Spec(x):
#     ecg_signal = x

#     # For simplicity, we'll just normalize the signal to have zero mean and unit variance.
#     ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

#     # Step 2: Compute the Spectrogram using Short-Time Fourier Transform (STFT)
#     n_fft = 56  # Number of FFT points
#     hop_length = 1  # Hop length in samples (controls the time resolution)
#     spectrogram = np.abs(librosa.stft(ecg_signal, n_fft=n_fft, hop_length=hop_length))

#     # Convert the magnitude spectrogram to dB scale
#     spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

#     # Step 3: Plot the Spectrogram
#     # plt.figure(figsize=(10, 6))
#     # librosa.display.specshow(spectrogram_db, sr=40, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='viridis')
#     # plt.colorbar(format='%+2.0f dB')
#     # plt.xlabel('Time (s)')
#     # plt.ylabel('Frequency (Hz)')
#     # plt.title('Spectrogram of ECG Signal')
#     # plt.ylim(0, 20)  # Adjust the frequency range for better visualization
#     # plt.show()

#     # stft = np.abs(librosa.stft(x, n_fft=2048,hop_length=256))
#     # stft = librosa.amplitude_to_db(stft, ref=np.max)
#     return spectrogram_db



# def spec_show(x):
#     fig = plt.figure(figsize=[10, 10])
#     plt.interactive(False)

#     ax = fig.add_subplot(111)
#     ax.axes.get_xaxis().set_visible(False)
#     ax.axes.get_yaxis().set_visible(False)
#     ax.set_frame_on(False)
#     x = librosa.display.specshow(x, y_axis='log', x_axis='time', ax=ax)
#     plt.show()

# base_path = '/home/livia/work/Recola2015/recordings_physio/'
# base_path_annotation = '/home/livia/work/Recola2015/ratings_gold_standard/al/'

# for pre_name in ['dev', 'train']:
#     file_names = glob.glob(os.path.join(base_path_annotation, pre_name + '*.csv'))
#     specs=[]
#     for file_name in sorted(file_names):
#         print(file_name)
#         path = os.path.join(base_path, file_name.split('/')[-1])#'/home/livia/work/Recola2015/recordings_physio/dev_2.csv'
#         anno_path = os.path.join(base_path_annotation, file_name)#'/home/livia/work/Recola2015/ratings_gold_standard/all/dev_2.csv'
#         anno_df = pd.read_csv(anno_path,sep=',',header=None)
#         anno_df.to_numpy()
#         vid_name,ts,_,lab = anno_df[0], anno_df[1], anno_df[2], anno_df[3]  #for arousal
#         lab=lab[1:-1].to_numpy()
#         ts=ts[1:-1].values
#         # print(lab)

#         all_anno={}
#         spec_dict={}
#         anno_dict=dict(zip(ts,lab))

#         # print(all_anno)

#         df = pd.read_csv(path,sep=';',header=None)
#         df.columns = ["time","EDA","ECG"]
#         df.to_numpy()
#         _,eda,ecg = df['time'], df['EDA'], df['ECG']
#         # times = [i for i in ts[1:]]
#         ecg=[float(x) for x in ecg[1:]]
#         e=np.array(ecg)

#         # specs=[]
#         num_specs=len(e)/40
#         for i in range(int(num_specs)):
#             spec=get_Spec(e[i:i+40])
#             specs.append(spec)
#             i=i+40

#         # spec_dict=dict(zip(ts,specs))

#         # print(specs)
#         # print(len(specs))
#     print('length of specs', len(specs), 'for', pre_name)
#     np.save(f'{pre_name+str(3)}.npy', specs)
