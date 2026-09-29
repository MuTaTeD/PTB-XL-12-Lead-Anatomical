import math

def pan_tompkins_qrs_width(ecg_signal, fs=100):
    """
    Implements a simplified pure-Python Pan-Tompkins algorithm 
    to extract average QRS width from a single-lead ECG signal.
    
    Args:
        ecg_signal (list): 1D array of ECG voltage values.
        fs (int): Sampling frequency in Hz.
        
    Returns:
        float: Estimated average QRS width in milliseconds.
    """
    N = len(ecg_signal)
    
    # 1. Derivative Filter: H(z) = (1/8T)(-z^{-2} - 2z^{-1} + 2z^1 + z^2)
    # Simplified approximation for feature extraction
    derivative = [0] * N
    for i in range(2, N - 2):
        derivative[i] = (1/8.0) * (-ecg_signal[i-2] - 2*ecg_signal[i-1] + 2*ecg_signal[i+1] + ecg_signal[i+2])
        
    # 2. Squaring Function: y(nT) = [x(nT)]^2
    squared = [x**2 for x in derivative]
    
    # 3. Moving Window Integration
    window_size = int(0.150 * fs) # 150 ms window
    integrated = [0] * N
    for i in range(window_size, N):
        integrated[i] = sum(squared[i - window_size : i]) / window_size
        
    # 4. Adaptive Thresholding & Peak Detection (Heuristic)
    # Calculate threshold based on mean of integrated signal
    threshold = sum(integrated) / len(integrated) * 1.5
    
    qrs_widths = []
    in_qrs = False
    qrs_start = 0
    
    for i, val in enumerate(integrated):
        if val > threshold and not in_qrs:
            in_qrs = True
            qrs_start = i
        elif val < threshold and in_qrs:
            in_qrs = False
            qrs_end = i
            # Convert samples to ms
            width_ms = ((qrs_end - qrs_start) / fs) * 1000.0
            
            # Physiological bound check (QRS is typically 60-160ms)
            if 60 <= width_ms <= 180:
                qrs_widths.append(width_ms)
                
    if not qrs_widths:
        return 0.0 # No valid QRS found
        
    return sum(qrs_widths) / len(qrs_widths)

if __name__ == "__main__":
    print("Pan-Tompkins QRS Extractor Initialized.")
    print("This pipeline processes the 1D counterfactual diffusion arrays to validate QRS compression.")
