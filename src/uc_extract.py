import os
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import scipy.interpolate

def filtUC(data, lowcut, fs=4, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = scipy.signal.butter(order, low, btype='lowpass')
    y = scipy.signal.filtfilt(b, a, data)
    return y


def uc_remove_baseline(sig, fs=4, width_in_min=10, show_plot=False):
    w = (fs * width_in_min * 60)//2
    
    all_baselines = []
    all_baselines.append([0, np.min(sig[:2*w])])
    for i in range(w, len(sig)-w, w):
        all_baselines.append([i, np.min(sig[i-w:i+w])])
    all_baselines.append([len(sig)-1, np.min(sig[2*w:])])
    
    f = scipy.interpolate.interp1d([x[0] for x in all_baselines], [x[1] for x in all_baselines], kind='cubic')
    baseline = f(np.arange(len(sig)))
    baseline_f = filtUC(baseline, 1.0/100, fs=fs, order=5)
    
    
    if show_plot:
        ts = np.arange(sig.shape[0])/fs
        tm = ts/60

        plt.figure(figsize=(12, 3))
        plt.title('Signal and Estimated Baseline')
        plt.plot(tm, sig)
        plt.scatter([tm[x[0]] for x in all_baselines], [x[1] for x in all_baselines], c='r')
        # plt.plot(tm, baseline)
        plt.plot(tm, baseline_f)
        # plt.xlim(0, 30)
        plt.show()

        plt.figure(figsize=(12, 3))
        plt.title('Signal After Baseline Removal')
        # plt.plot(tm, sig - baseline)
        plt.plot(tm, sig - baseline_f)
        # plt.xlim(0, 30)
        plt.show()
        
    return sig - baseline_f


def find_raw_peaks(sig, offset=1):
    """Find peaks based on local minima based on first derivitive zero crossings"""
    sig_d = np.diff(sig)
    mask = np.logical_and(sig_d[:-1] > 0, sig_d[1:] <= 0)
    idx_peaks = [i+offset for i, v in enumerate(mask) if v]
    return idx_peaks


def filter_peaks_window(sig, idx_peaks, min_spacing=90, fs=4):
    """Select only those peaks that are local maxima +/- half minimum UC to UC spacing"""
    w = int(min_spacing*fs//2)
    
    # compute actual local minima for each peak, only keep actual local minima
    filt_peaks = [(i, np.argmax(sig[max(0, i-w):i+w])+max(0, i-w) )  for i in idx_peaks ]
    filt_peaks = [idx_actual for idx, idx_actual in filt_peaks 
                  if abs(idx -idx_actual) <= 1]
    
    # handle potential false peak at start of recording
    if len(filt_peaks) > 0 and filt_peaks[0] < w:
        filt_peaks = filt_peaks[1:]
    return filt_peaks


def find_uc_gaps(idx_peaks, sig):
    """Find minima between identified peaks"""
    idx_min = [np.argmin(sig[idx_peaks[i-1]: idx_peaks[i]])+idx_peaks[i-1] 
               for i in range(1, len(idx_peaks))]
    idx_min = ([np.argmin(sig[: idx_peaks[0]])] 
               + idx_min 
               + [np.argmin(sig[idx_peaks[-1]:])+idx_peaks[-1]])
    return idx_min


def extract_uc_features(idx_peaks, sig, ts):
    """Characterize each UC candidate based on onset and release"""
    idx_min = find_uc_gaps(idx_peaks, sig)
    annotated_peaks = []
    for i in range(len(idx_peaks)):
        delta_mag_l = sig[idx_peaks[i]]-sig[idx_min[i]]
        delta_mag_r = sig[idx_peaks[i]]-sig[idx_min[i+1]]
        
        delta_min = min(delta_mag_l, delta_mag_r)
        delta_max = max(delta_mag_l, delta_mag_r)
        
        delta_min_rel = delta_min/sig[idx_peaks[i]]
        delta_max_rel = delta_max/sig[idx_peaks[i]]
                  
        entry = {
            'idxPeak':idx_peaks[i], 'tPeak':ts[idx_peaks[i]]/60,
            'delta_min':delta_min, 'delta_min_rel':delta_min_rel,
            'delta_max':delta_max, 'delta_max_rel':delta_max_rel,
            'idxMinL':idx_min[i], 'delta_mag_l':delta_mag_l, 
            'delta_rel_mag_l':delta_mag_l/sig[idx_peaks[i]],
            'idxMinR':idx_min[i+1], 'delta_mag_r':delta_mag_r, 
            'delta_rel_mag_r':delta_mag_r/sig[idx_peaks[i]],
        }
        annotated_peaks.append(entry)

    return annotated_peaks


def filter_uc_singleton_mag(idx_peaks, sig, ts,
                            thesh_abs_min_mag=10,thesh_rel_min_mag=.3):
    
    print('Remove Peaks - Singleton using Mag Change')
    while True:
        annotated_peaks = extract_uc_features(idx_peaks, sig, ts)
        idx_peaks = [entry['idxPeak'] for entry in annotated_peaks
                     if entry['delta_max'] > thesh_abs_min_mag 
                     and entry['delta_max_rel'] > thesh_rel_min_mag]

        if len(idx_peaks) == len(annotated_peaks):
            return idx_peaks

        ignored = [entry for entry in annotated_peaks
                   if not(entry['delta_max'] > thesh_abs_min_mag 
                   and entry['delta_max_rel'] > thesh_rel_min_mag)]
        for entry in ignored:
            print('    Peak:    @ {:5.1f} min   Mag: {:5.1f}'.format(
                entry['tPeak'], sig[entry['idxPeak']]))
            
 
def filter_uc_adjacent_mag(idx_peaks, sig, ts,
                           thesh_abs_min_mag=10,thesh_rel_min_mag=.3):
    
    print('Remove Peaks - Adjacent using Mag')
    while True:
        annotated_peaks = extract_uc_features(idx_peaks, sig, ts)
     
        filtered_peaks = []
        i = 0
        while i < len(annotated_peaks):
            entry = annotated_peaks[i]
            if (entry['delta_min'] > thesh_abs_min_mag 
                and entry['delta_min_rel'] > thesh_rel_min_mag):
                filtered_peaks.append(entry)
                i+=1
                continue
            
            if (entry['delta_mag_l'] < thesh_abs_min_mag 
                or entry['delta_rel_mag_l'] < thesh_rel_min_mag):
                # print('** Left below threshold')
                left_entry = filtered_peaks[-1]
                filtered_peaks = filtered_peaks[:-1]
                right_entry = entry
                i+= 1

            elif (entry['delta_mag_r'] < thesh_abs_min_mag 
                  or entry['delta_rel_mag_r'] < thesh_rel_min_mag):
                left_entry = entry
                right_entry = annotated_peaks[i+1]
                i+= 2
            else:
                assert False

            
            # combine entries
            if sig[left_entry['idxPeak']] >= sig[right_entry['idxPeak']]:
                # keep left peak - fix-up right params
                print('    Peak:    @ {:5.1f} min   Mag: {:5.1f}'.format(
                    right_entry['tPeak'], sig[right_entry['idxPeak']]))
                new_entry = left_entry     
                mag = sig[new_entry['idxPeak']]
                
                idx_min = right_entry['idxMinR']
                mag_min = sig[idx_min]
                
                new_entry['idxMinR'] = idx_min
                new_entry['delta_mag_r'] = mag - mag_min
                new_entry['delta_rel_mag_r'] = new_entry['delta_mag_r'] / mag

            else:
                # keep right peak - fix-up right params
                print('    Peak:    @ {:5.1f} min   Mag: {:5.1f}'.format(
                    left_entry['tPeak'], sig[left_entry['idxPeak']]))

                new_entry = right_entry     
                mag = sig[new_entry['idxPeak']]
                
                idx_min = left_entry['idxMinL']
                mag_min = sig[idx_min]
                
                new_entry['idxMinL'] = idx_min
                new_entry['delta_mag_l'] = mag - mag_min
                new_entry['delta_rel_mag_l'] = new_entry['delta_mag_l'] / mag

            new_entry['delta_min'] = min(new_entry['delta_mag_l'], 
            							 new_entry['delta_mag_r'])
            new_entry['delta_max'] = max(new_entry['delta_mag_l'], 
            							 new_entry['delta_mag_r'])
            new_entry['delta_min_rel'] = min(new_entry['delta_rel_mag_l'], 
            								 new_entry['delta_rel_mag_r'])
            new_entry['delta_max_rel'] = max(new_entry['delta_rel_mag_l'], 
            								 new_entry['delta_rel_mag_r'])
                
            filtered_peaks.append(new_entry)

        # extract peaks and determine whether there have been any reductions this iteration
        # stop when no further changes
        idx_peaks = [entry['idxPeak'] for entry in filtered_peaks]
        
        if len(annotated_peaks) == len(idx_peaks):
            return idx_peaks


def find_uc_peaks(sig, ts, min_spacing=90, fs=4,
                         thesh_abs_min_mag=8,thesh_rel_min_mag=.3):

    idx_peaks = find_raw_peaks(sig)
    idx_peaks = filter_peaks_window(sig, idx_peaks, min_spacing=90, fs=4)
    
    idx_peaks = filter_uc_singleton_mag(idx_peaks, sig, ts,
                                        thesh_abs_min_mag=thesh_abs_min_mag,
                                        thesh_rel_min_mag=thesh_rel_min_mag)
    
    idx_peaks = filter_uc_adjacent_mag(idx_peaks, sig, ts,
                                       thesh_abs_min_mag=thesh_abs_min_mag,
                                       thesh_rel_min_mag=thesh_rel_min_mag)
        
    return idx_peaks



