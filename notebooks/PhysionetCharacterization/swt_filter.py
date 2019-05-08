import pywt
import numpy as np

def swt_align(sig, levels=0, include_extra_padding=True):
    if levels == 0:
        return sig, 0, 0
    
    modulus = 2**levels
    excess = len(sig) % modulus
    n_pad = modulus-excess
    
    if include_extra_padding:
        n_pad += modulus
        
    n_pad_l = n_pad // 2
    n_pad_r = n_pad - n_pad_l
    if n_pad != 0:  
        sig = np.pad(sig, (n_pad_l, n_pad_r), 'edge')

    return sig, n_pad_l, n_pad_r

    
def swt_filter(coeffs, exclude_level=0, exclude_level_d=None, wavelet='db4'):

    if exclude_level_d is None:
        exclude_level_d = exclude_level

    coeffs = [[cA, cD] for cA, cD in coeffs]
    
    for j in range(exclude_level, len(coeffs)):
        coeffs[j][1] = coeffs[j][1]*0
        
        
    for j in range(exclude_level_d, len(coeffs)):
        if j != 0:  
            coeffs[j][0] = coeffs[j][0]*0
    
    baseline = pywt.iswt(coeffs, wavelet)
    return baseline    
    
    
def swt_process_signal(sig, exclude_level=0, baseline_exclude_level=0, 
                       total_levels=11, fs=1, **kwargs):
    
    len_orig = len(sig)
    sig, n_pad_l, n_pad_r = swt_align(sig, total_levels)
    coeffs = pywt.swt(sig, 'db4', level=total_levels)
    
    baseline = swt_filter(coeffs, baseline_exclude_level)
    sig_f = swt_filter(coeffs, exclude_level)
    delta = sig_f - baseline

    if n_pad_r > 0:
        baseline, sig_f = baseline[n_pad_l:-n_pad_r], sig_f[n_pad_l:-n_pad_r]
    elif n_pad_l > 0:
        baseline, sig_f = baseline[n_pad_l:], sig_f[n_pad_l:]
        
    ts = np.arange(len(baseline))/fs
    return baseline, sig_f, ts    


def swt_process_uc_signal(sig, exclude_detail_level=0, exclude_before=2,
                          total_levels=11, fs=1, wavelet='db4', should_clip=True):
    
    len_orig = len(sig)
    sig, n_pad_l, n_pad_r = swt_align(sig, total_levels)
    coeffs = pywt.swt(sig, wavelet, level=total_levels)
    coeffs = [[cA, cD] for cA, cD in coeffs]
    
    for j in range(exclude_before):
            coeffs[j][0] = coeffs[j][0]*0
            coeffs[j][1] = coeffs[j][1]*0
            
    for j in range(exclude_detail_level, len(coeffs)):
            coeffs[j][0] = coeffs[j][0]*0
            coeffs[j][1] = coeffs[j][1]*0
    
    sig_f = pywt.iswt(coeffs, wavelet)

    
    if n_pad_r > 0:
        sig_f = sig_f[n_pad_l:-n_pad_r]
    elif n_pad_l > 0:
        sig_f = sig_f[n_pad_l:]
        
        
    if should_clip:
        sig_f[sig_f < 0] = 0
        
    ts = np.arange(len(sig_f))/fs
    return sig_f, ts 


def replace_invalid_values(sig, window=60, use_interpolate=False, **kwargs):
    """Replace outlier values with neighborhood median or via interpolation of adjacent values"""
    if use_interpolate:
        mask = sig > 0
        idx = np.arange(len(sig))
        
        new_sig = np.interp(idx, idx[mask], sig[mask])
    else:                         # use local median
        new_sig = np.copy(sig)
        for i, v in enumerate(sig):
            if v != 0:
                continue
            new_sig[i] = get_replacement_value(i, sig, window, **kwargs) 
    return new_sig
    
    
def get_replacement_value(i, sig, window=60, min_valid_samples=10):
    """Find median value within window, grow window if insufficient values"""
    half = window // 2
    if i < half:
        seg = sig[0:window]
    elif  i+half > len(sig):
        seg = sig[-window:]
    else:
        seg = sig[i-half:i+half]
    seg = seg[seg != 0]
    
    if len(seg) >= min_valid_samples:
        return np.median(seg) 
    else:
        return get_replacement_value(i, sig, window*2)  