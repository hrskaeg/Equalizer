import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


bands = [(20,70), (70,140), (140, 280), (280, 630), (630, 1600), (1600, 4000), (4000, 10000), (10000, 20000)] #Hardcodede bånd som normalt bruges af musik equalizere

def open_file(file_path):
    fs, x = wavfile.read(file_path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(float)
    x = x /np.max(np.abs(x))
    return x, fs

# Helper fra miniprojekt B
def energi(seg, fs, f_lav, f_hoj):
    N = len(seg)
    X = np.fft.fft(seg)
    df = fs / N                     # frekvensopløsning

    m_lav = round(f_lav / df)         # bin-nummer for nedre grænse
    m_hoj = round(f_hoj / df)         # bin-nummer for øvre grænse

    E = 0
    for m in range(m_lav, m_hoj):   # summer over binsene i båndet
        E += abs(X[m]) ** 2         # |X(m)|²

    return 2 * E / N                # ganget med 2/N



# Opgave 1


def design_filters(fs, numtaps, metode="firwin"):
    filters = []
    for i in range(len(bands)):
        low, high = bands[i] # Freq cutoffs
        if metode == "firwin":
            b = signal.firwin(numtaps, [low, high], pass_zero=True, fs=fs) # generate FIR
        if metode == "frekvenssampling":
            b = frekvenssampling_bandpass(low, high, fs, numtaps)
        filters.append(b)
    return filters


# Manuel frekvenssampling krav fra opgave
def frekvenssampling_bandpass(f_lav, f_høj, fs, M):
    # Knækfrekvenser (f_c) -> bin-numre. 2 knæfkrekvenser da det er båndpas
    m_lav = round(M * f_lav / fs)
    m_høj = round(M * f_høj / fs)

    # Bin-værdier på området (Op til fs/2)
    H = np.zeros(M)
    # +1 for at få den 'sidste' bin med ved ulige M. Hvis m er på området, lad frekvensen gå igennem 
    for m in range(M // 2 + 1): 
        if m_lav <= m <= m_høj:
            H[m] = 1

    # Konstruer H(m) ved spejling (starter ved index 1, da 0 HZ DC er der)
    for m in range(1, M // 2 + 1):
        H[M - m] = H[m]         # index m spejlings_index = M - n

    # IDFT for at få impulsresponsen
    h = np.real(np.fft.ifft(H)) # real for at fjerne evt. imaginære rester (som slides foreslår)
    h = np.fft.fftshift(h)      # centrer filteret fra (n= 0) til midten af arrayet (n=M-1/2) (som slides foreslår)

    # gang med vinduesfunktionen (hamming valgt)
    h = h* np.hamming(M)
    return h

    

# Opgave 2
def frekvenskarakteristik(h, fs, N_fft=65536):
    H = np.fft.fft(h, N_fft)                    # H(m) = DFT af h(n), passer n=N_fft for at padde med zeros (smoothness)
    k = N_fft // 2                                # bins er 0 til fs/2
    f = np.arange(k) * fs / N_fft               # frekvensaksen
    H_dB = 20 * np.log10(np.abs(H[:k]) + 1e-12) # H_dB = forstærkning ved hver frekvens i dB (+1e-12 undgår log(0)) 
    return f, H_dB


def plot_frekvenskarakteristik(filters, fs, titel):
    plt.figure(figsize=(10, 4))
    for i in range(len(filters)):
        f, H_dB = frekvenskarakteristik(filters[i], fs)
        low, high = bands[i]
        plt.plot(f, H_dB, linewidth=0.8, label=f"{low}-{high} Hz")
    plt.xscale("log")
    plt.xlim(10, fs / 2)
    plt.ylim(-80, 5)
    plt.title(f"Frekvenskarakteristik - {titel}")
    plt.xlabel("frekvens [Hz]")
    plt.ylabel("|H(m)| [dB]")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.show()


def plot_impulsrespons(h, titel):
    plt.figure(figsize=(10, 3))
    plt.plot(np.arange(len(h)), h, linewidth=0.8)
    plt.title(f"Impulsrespons h(n) - {titel}")
    plt.xlabel("n [samples]")
    plt.ylabel("h(n)")
    plt.grid(alpha=0.3)
    plt.show()



def apply_filter(x, filters, gains):
    y = np.zeros(len(x))
    for i in range(len(filters)):
        g = 10 ** (gains[i] / 20) # dB -> linear conversion
        y = y + g * signal.lfilter(filters[i], 1, x) # Filter, scale, add
    return y



# Opgave 3
def samlet_impulsrespons(filters, gains, N):
    impuls = np.zeros(N)    # N = numptaps for at få hele filteret med
    impuls[0] = 1           # vores impulssignal δ(n) (som returnerer filterets koefficienter)
    return (apply_filter(impuls, filters, gains))


