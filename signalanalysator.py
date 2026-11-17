import numpy as np
import matplotlib.pyplot as plt


def plot_tid(x, fs, t_start, varighed):        # punkt 1
    n0 = int(t_start * fs)            # seconds → first sample number
    N  = int(varighed * fs)           # seconds → number of samples
    seg = x[n0 : n0 + N]              # cut out that piece of the list
    t = (n0 + np.arange(N)) / fs      # time of each sample, in seconds

    plt.figure(figsize=(10, 3))
    plt.plot(t, seg, linewidth=0.5)
    plt.xlabel("tid [s]")
    plt.ylabel("Amplitude")
    plt.show()

    return seg                        # hand the piece on to spektrum(), energi() etc.

def spektrum(seg, fs, titel="Signal"):                           # punkt 2
    N = len(seg)
    X = np.fft.fft(seg)

    K = N // 2                  # kun værdier fra 0 til fs/2
    f = np.arange(K) * fs / N   # frekvensakse
    A = np.abs(X[:K]) / N       # Amplitude på området (normalisering med N)
    A[1:] *= 2

    print(f"{titel}: N = {N}, frekvensopløsning = {fs / N:.4f} Hz")

    plt.figure(figsize=(10, 3))
    y = A
    plt.plot(f, y, linewidth=0.5)
    plt.xlim(right= fs / 2) # grænser for x
    plt.title(f"{titel} - amplitudespektrum")
    plt.xlabel("frekvens [Hz]")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    plt.show()
 
    return f, A


def udglat(f, A, bredde_Hz, titel):
    A_glat = np.convolve(A, w, mode="same")¨
    df = f[1] - f[0]                         # frekvensopløsning
    L = max(1, int(bredde_Hz / df))          # vinduets bredde i antal bins
    halv = L // 2                            # antal naboer på hver side

    A_glat = np.zeros(len(A))
    for m in range(len(A)):                  # vinduet glider hen over spektret
        start = max(0, m - halv)             # vinduets venstre kant
        slut = min(len(A), m + halv + 1)     # vinduets højre kant
        A_glat[m] = np.mean(A[start:slut])   # gennemsnit af bins i vinduet

    plt.figure(figsize=(10, 3))
    plt.plot(f, A, linewidth=0.5, alpha=0.5, label="uden udglatning")
    plt.plot(f, A_glat, linewidth=1.2, label=f"udglattet ({bredde_Hz} Hz)")
    plt.title(f"{titel} - udglattet spektrum")
    plt.xlabel("frekvens [Hz]")
    plt.ylabel("Amplitude")
    plt.legend()
    
    return A_glat

def energi(seg, fs, f_lav, f_hoj):
    N = len(seg)
    X = np.fft.fft(seg)
    df = fs / N                     # frekvensopløsning

    m_lav = int(f_lav / df)         # bin-nummer for nedre grænse
    m_hoj = int(f_hoj / df)         # bin-nummer for øvre grænse

    E = 0
    for m in range(m_lav, m_hoj):   # summer over binsene i båndet
        E += abs(X[m]) ** 2         # |X(m)|²

    return 2 * E / N                # ganget med 2/N

def spektrogram(seg, fs, vindue_s, titel):
    N = len(seg)                             # antal samples i hele signalet

    L = int(vindue_s * fs)                   # segmentlængde i samples
    overlap = 50                             # γ: overlap i procent
    hop = int(L * (1 - overlap / 100))       # afstand mellem to segmenters start
    K = int(N / L * 1 / (1 - overlap / 100) - 1)   # antal segmenter
    w = np.hanning(L)                        # vinduet w(n)

    # Tom tabel: én række pr. frekvens, én søjle pr. segment
    S = np.zeros((L // 2, K))

    # Én DFT pr. segment
    for k in range(K):
        start = k * hop                      # hvor segment k starter
        stykke = seg[start : start + L]      # klip segmentet ud: x(p)
        stykke = stykke * w                  # gang med vinduet: x(p)·w(n)
        X = np.fft.fft(stykke)               # DFT af segmentet
        S[:, k] = np.abs(X[:L // 2])         # gem størrelsen (0 til fs/2) som søjle k

    # Akser
    f = np.arange(L // 2) * fs / L           # frekvens for hver række, Δf = fs/L
    t = (np.arange(K) * hop + L / 2) / fs    # tid for hver søjle (segmentets midte)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 20 * np.log10(S + 1e-12), shading="auto")
    plt.colorbar(label="Amplitude [dB]")
    plt.title(f"{titel} - spektrogram")
    plt.xlabel("tid [s]")
    plt.ylabel("frekvens [Hz]")

    return t, f, S