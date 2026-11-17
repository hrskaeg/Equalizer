import Equalizer as eq
import time
import sounddevice as sd


gains = [0,0,0,0,0,0,0,0] # gains i dB
numtaps = 4095 # FIR filter length (ulige tal = type 1 FIR filter)

file_path = "C:/Users/Mads/repos/skole/Equalizer/Master - Losing Game (Main) [16 bit].wav"



x, fs = eq.open_file(file_path)


# Opgave 1 og 2

# Opdeler signaler i 8 båndpasfiltre (båndpas grænser defineret i Equalizer.py)
filters_firwin = eq.design_filters(fs, numtaps, 'firwin')
filters_fs = eq.design_filters(fs, numtaps, 'frekvenssampling')

# Frekvenskarakteristik for alle 8 båndpasfiltre
eq.plot_frekvenskarakteristik(filters_firwin, fs, 'firwin')
eq.plot_frekvenskarakteristik(filters_fs, fs, 'frekvenssampling')

#impulsrespons for et filter (bånd 4 valgt = 1000-3000 Hz)
eq.plot_impulsrespons(filters_fs[3], 'impulsrespons, 1000-3000 Hz, frekvenssampling')


# Opgave 4
indstillinger = {
    "Bas-boost":        [6, 6, 3, 0, 0, 0, 0, 0],
    "Vocal-boost":      [0, 0, 0, 0, 0, 6, 0, 0],
    "Lineær":           [-12, -9, -6, -3, 0, 3, 6, 9],
    "whatever":         [-18, 18, -18, 18, -18, 18, -18, 18],
}

for navn, g in indstillinger.items():
    h_tot = eq.samlet_impulsrespons(filters_fs, g, numtaps)
    eq.plot_impulsrespons(h_tot, navn)


# Original
sd.play(x,fs)
time.sleep(10)
sd.stop()

for navn, g in indstillinger.items():
    y = eq.apply_filter(x, filters_fs, g) 
    sd.play(y, fs)
    time.sleep(10)
    sd.stop()
    