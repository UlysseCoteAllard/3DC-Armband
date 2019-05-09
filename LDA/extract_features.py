from scipy import stats
from scipy.fftpack import rfft, irfft
from pylab import *
from nitime.algorithms import autoregressive
import pywt
import math
import sampen.sampen2
from scipy import signal


def cmp(a, b):  # easy port from Python 2 to Python 3
    if a < b:
        return -1
    if a > b:
        return 1
    else:
        return 0


def zero_crossing(vector, threshold=0.1):
    number_zero_crossing = 0
    current_sign = cmp(vector[0], 0)
    for i in range(0, len(vector)):
        if current_sign == -1:
            if current_sign != cmp(vector[i], threshold):  # We give a delta to consider that the zero was crossed
                current_sign = cmp(vector[i], 0)
                number_zero_crossing += 1
        else:
            if current_sign != cmp(vector[i], -threshold):
                current_sign = cmp(vector[i], 0)
                number_zero_crossing += 1
    return number_zero_crossing


def iemg(vector):
    iemg_value = 0.0
    for value in vector:
        iemg_value += abs(value)
    return iemg_value


def waveform_length(vector):
    wl_sum = 0.0
    for i in range(0, len(vector)-1):
        wl_sum += abs(vector[i+1] - vector[i])
    return wl_sum


def slop_sign_change(vector, threshold=0.1):
    slope_change = 0
    for i in range(1,len(vector)-1):
        get_x = (vector[i]-vector[i-1])*(vector[i]-vector[i+1])
        if(get_x >= threshold):
            slope_change += 1
    return slope_change


def skewness(vector):
    return stats.skew(vector)


def rms(vector):
    total_sum = 0.0
    for i in range(0, len(vector)):
        total_sum += vector[i]*vector[i]
    sigma = math.sqrt(total_sum/(len(vector)))
    return sigma


def integratedAbsoluteValue(vector):
    total_sum = 0.0
    for i in range(0, len(vector)):
        total_sum += abs(vector[i])
    return total_sum/len(vector)


def autoregressiveCoefficient(vector, nmbr_of_coefficient=11):
    AR_coeff, innovations_variance = autoregressive.AR_est_YW(np.array(vector), nmbr_of_coefficient)
    return AR_coeff


def cepstral_coefficients(vector, order=4):
    ar_coeff = autoregressiveCoefficient(vector, nmbr_of_coefficient=order)
    cc = np.zeros(order)
    for index in range(order):
        if index == 0:
            cc[index] = -1.*ar_coeff[index]
        else:
            cc_to_add = 0.
            for i in range(index):
                cc_to_add += (1.-(float(i+1)/float(index+1)))*ar_coeff[index]*cc[index-i]
            cc[index] = -1.*ar_coeff[index]-cc_to_add
    return cc


def mDWT_NinaPro_direct_implementation(vector, level=3, wavelet='db7'):
    coefficients = pywt.wavedec(vector, level=level, wavelet=wavelet)
    C = []
    for vector in coefficients:
        C.extend(vector)
    N = len(C)
    SMax = int(math.log(N, 2))
    Mxk = []
    for s in range(SMax):
        CMax = int(round((N/(2.**(s+1)))-1))
        Mxk.append(np.sum(np.abs(C[0:(CMax)])))
    return Mxk


def mDWT_NinaPro(vector, level=3, wavelet='db7'):
    coefficients = pywt.wavedec(vector, level=level, wavelet=wavelet)
    approx = coefficients[0]
    N = len(approx)
    SMax = int(math.log(N, 2))
    Mxk = []
    for s in range(SMax):
        CMax = int(round((N/(2.**(s+1)))-1))
        Mxk.append(np.sum(np.abs(approx[0:(CMax)])))
    return Mxk


def mDWT(vector, level=3, wavelet='db7'):
    Mxk = []
    coefficients = pywt.wavedec(vector, level=level, wavelet=wavelet)

    N = len(vector)
    for s in range(len(coefficients)):
        # s+1 because we start at 0 instead of 1.
        upper_bound_u = int(round((N/2**(s+1)) - 1))
        # -1 because we start at 0 instead of 1
        Mxk.append(np.sum(np.abs(coefficients[s][0:(upper_bound_u-1)])))
    return Mxk


def HIST(vector, threshold_nmbr_of_sigma, nmbr_bins=20):
    # calculate sigma of signal
    sigma = np.std(vector)
    mean = np.std(vector)
    threshold = threshold_nmbr_of_sigma*sigma
    hist, bin_edges = np.histogram(vector, bins=nmbr_bins, range=(mean-threshold, mean+threshold))
    return hist


def Hjorth_activity_parameter(vector):
    return np.var(vector)


def Hjorth_mobility_parameter(vector):
    first_derivative = np.diff(vector)
    ratio = np.var(first_derivative, ddof=1)/np.var(vector, ddof=1)#Sample variance
    return math.sqrt(ratio)


def Hjorth_complexity_parameter(vector):
    mobility_signal = Hjorth_mobility_parameter(vector)
    mobility_first_derivate = Hjorth_mobility_parameter(np.diff(vector))
    return mobility_first_derivate/mobility_signal


def MAV(vector):
    total_sum = 0.0
    for i in range(len(vector)):
        total_sum += abs(vector[i])
    return total_sum/len(vector)


def MAV1(vector):
    vector_array = np.array(vector)
    total_sum = 0.0
    for i in range(0,len(vector_array)):
        if((i+1) < 0.25*len(vector_array) or (i+1) > 0.75*len(vector_array)):
            w = 0.5
        else:
            w = 1.0
        total_sum += abs(vector_array[i]*w)
    return total_sum/len(vector_array)


def MAV2(vector):
    total_sum = 0.0
    vector_array = np.array(vector)
    for i in range(0,len(vector_array)):
        if((i+1) < 0.25*len(vector_array)):
            w = ((4.0*(i+1))/len(vector_array))
        elif((i+1) > 0.75*len(vector_array)):
            w = (4.0*((i+1)-len(vector_array)))/len(vector_array)
        else:
            w = 1.0
        total_sum += abs(vector_array[i]*w)
    return total_sum/len(vector_array)


def Willison_amplitude(vector, threshold=0.1):
    wamp_decision = 0
    for i in range(1, len(vector)):
        get_x = abs(vector[i]-vector[i-1])
        if(get_x >= threshold):
            wamp_decision += 1
    return wamp_decision


def real_cepstrum(x, n=None):
    x_abs = []
    for i in x:  # Rectify the signal
        x_abs.append(abs(i))
    fft_resuts = rfft(x)
    fft_abs = []
    for i in fft_resuts:
        fft_abs.append(abs(i+0.000000001))
    log_fft = np.log(fft_abs)
    inv_fft = irfft(log_fft)
    y = real(inv_fft)
    return y


def time_domain_statistics(vector):
    features = []
    features.append(MAV(vector))
    features.append(zero_crossing(vector))
    features.append(slop_sign_change(vector))
    features.append(waveform_length(vector))

    return features

def get_article_8_electrodes_features(vector):
    features = []
    features.append(zero_crossing(vector, threshold=1))
    features.append(waveform_length(vector))
    features.append(slop_sign_change(vector, threshold=1))
    features.append(skewness(vector))
    features.append(rms(vector))
    features.append(MAV(vector))
    features.append(integratedAbsoluteValue(vector))
    features.extend(autoregressiveCoefficient(vector, 11))
    features.append(Hjorth_activity_parameter(vector))
    features.append(Hjorth_mobility_parameter(vector))
    features.append(Hjorth_complexity_parameter(vector))

    return features

def SampEn(vector, m=2, r_multiply_by_sigma=.2):
    r = r_multiply_by_sigma*np.std(vector)
    results = sampen.sampen2(data=vector.tolist(), mm=m, r=r, normalize=True)
    results_SampEN = []
    for x in np.array(results)[:, 1]:
        if x is not None:
            results_SampEN.append(x)
        else:
            results_SampEN.append(-100.)
    return list(results_SampEN)


def SampEn_4_pipeline(vector):
    features = SampEn(vector)
    features.append(rms(vector))
    features.append(waveform_length(vector))
    features.extend(cepstral_coefficients(vector, order=4))
    return features

def NinaPro_best(vector):
    features = time_domain_statistics(vector)
    features.extend(HIST(vector, threshold_nmbr_of_sigma=3, nmbr_bins=20))
    features.extend(mDWT_NinaPro_direct_implementation(vector, wavelet='db7', level=3))
    return np.array(features, dtype=np.float32)

def NinaPro_normalized_combination_best(vector):
    features = time_domain_statistics(vector)
    features.extend(HIST(vector, threshold_nmbr_of_sigma=3, nmbr_bins=20))
    features.extend(mDWT_NinaPro_direct_implementation(vector, wavelet='db7', level=3))
    return np.array(features, dtype=np.float32)

def get_special_features(vector):
    features = []
    features.append(skewness(vector))
    features.append(rms(vector))
    features.append(MAV(vector))
    features.append(integratedAbsoluteValue(vector))
    features.extend(autoregressiveCoefficient(vector, 11))
    features.append(Hjorth_activity_parameter(vector))
    features.append(Hjorth_mobility_parameter(vector))
    features.append(Hjorth_complexity_parameter(vector))
    return features

def format_dataset(dataset, method='article_8'):
    dataset_to_return = []
    for example in dataset:
        example_formatted = []
        for vector_electrode in example:
            # Do high pass filter
            if method == 'article_8':
                example_formatted.append(get_article_8_electrodes_features(vector_electrode))
            elif method == 'sampen':
                example_formatted.append(SampEn(vector_electrode))
            elif method == 'sampen_4':
                example_formatted.append(SampEn_4_pipeline(vector_electrode))
            elif method == 'NinaPro_best':
                example_formatted.append(NinaPro_best(vector_electrode))
            elif method == 'mDWT_only':
                example_formatted.append(mDWT_NinaPro_direct_implementation(vector=vector_electrode))
            elif method == 'TD_stats':
                example_formatted.append(time_domain_statistics(vector_electrode))
            else:
                example_formatted.append(time_domain_statistics(vector_electrode))
        dataset_to_return.append(np.array(example_formatted).transpose().flatten())
    return dataset_to_return

def show_signal_to_transform(emg_vector):
    import matplotlib.pyplot as plt
    plt.plot(emg_vector)
    plt.show()

def butter_highpass(lowcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    sos = signal.butter(N=order, Wn=low, btype='highpass', output='sos')
    return sos

def butter_highpass_filter(data, lowcut, fs, order=4):
    sos = butter_highpass(lowcut, fs, order=order)
    y = signal.sosfilt(sos, data)
    return y

if __name__ == '__main__':
    array = np.arange(1, 250)
    print(array)
    print(mDWT(array, wavelet='db7', level=3))
    print(mDWT_NinaPro_direct_implementation(array, wavelet='db7', level=3))
