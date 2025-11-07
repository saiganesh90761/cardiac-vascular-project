"""
Electro-Mechanical Heart Simulator

This module provides a complete software-based electro-mechanical heart model
that generates ECG signals (electrical activity) and corresponding mechanical
contraction signals based on the ECG.

Author: Cardiac Project
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import math
import random
import csv
from datetime import datetime


# ============================================================================
# ECG Signal Generation
# ============================================================================

def generate_ecg(
    condition: str = "normal",
    duration: float = 10.0,
    sampling_rate: int = 250,
    noise_level: float = 0.01
) -> Tuple[np.ndarray, float]:
    """
    Generate ECG signal based on heart condition.
    
    Parameters
    ----------
    condition : str
        Heart condition: "normal", "tachycardia", "bradycardia", "irregular"
    duration : float
        Duration of ECG signal in seconds (5-20 seconds recommended)
    sampling_rate : int
        Sampling rate in Hz (default: 250 Hz)
    noise_level : float
        Standard deviation of Gaussian noise (default: 0.01)
    
    Returns
    -------
    Tuple[np.ndarray, float]
        ECG signal array and calculated heart rate (bpm)
    """
    # Define heart rates for different conditions
    heart_rate_map = {
        "normal": 75.0,      # Normal: 60-100 bpm, use 75 as average
        "tachycardia": 120.0, # Tachycardia: >100 bpm
        "bradycardia": 50.0,  # Bradycardia: <60 bpm
        "irregular": 75.0     # Irregular: variable rate around 75
    }
    
    base_hr = heart_rate_map.get(condition.lower(), 75.0)
    
    # Generate time array
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Calculate RR interval (time between beats)
    rr_interval = 60.0 / max(base_hr, 1.0)  # seconds per beat
    
    # Initialize ECG signal
    ecg = np.zeros(n_samples)
    
    # Define ECG waveform components (Gaussian-based model)
    def gaussian_wave(t, mu, sigma, amplitude):
        """Generate Gaussian waveform component."""
        return amplitude * np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    
    # ECG waveform parameters (relative to beat start, in seconds)
    # P wave: atrial depolarization
    P = {"mu": 0.10, "sigma": 0.025, "amp": 0.15}
    # Q wave: initial ventricular depolarization
    Q = {"mu": 0.18, "sigma": 0.010, "amp": -0.20}
    # R wave: main ventricular depolarization (largest peak)
    R = {"mu": 0.20, "sigma": 0.015, "amp": 1.0}
    # S wave: end of ventricular depolarization
    S = {"mu": 0.22, "sigma": 0.012, "amp": -0.30}
    # T wave: ventricular repolarization
    T = {"mu": 0.40, "sigma": 0.04, "amp": 0.40}
    
    # Generate beat start times
    beat_starts = []
    current_time = 0.0
    
    while current_time < duration + rr_interval:
        if condition.lower() == "irregular":
            # Irregular rhythm: variable RR intervals
            jitter = random.gauss(0.0, 0.15)  # Larger jitter for irregularity
            variability = random.uniform(-0.3, 0.3)  # ±30% variation
            rr_current = rr_interval * (1.0 + variability)
        else:
            # Normal variability (respiratory sinus arrhythmia)
            jitter = random.gauss(0.0, 0.03)  # Small jitter
            rr_current = rr_interval
        
        beat_starts.append(max(0.0, current_time + jitter))
        current_time += rr_current
    
    # Generate ECG signal by summing contributions from all beats
    for i, time_point in enumerate(t):
        signal_value = 0.0
        
        # Sum contributions from nearby beats (check last 5 beats for efficiency)
        for beat_start in beat_starts[-5:]:
            dt = time_point - beat_start
            
            # Only consider beats within one RR interval
            if -0.2 <= dt <= 0.8:
                signal_value += (
                    gaussian_wave(dt, P["mu"], P["sigma"], P["amp"]) +
                    gaussian_wave(dt, Q["mu"], Q["sigma"], Q["amp"]) +
                    gaussian_wave(dt, R["mu"], R["sigma"], R["amp"]) +
                    gaussian_wave(dt, S["mu"], S["sigma"], S["amp"]) +
                    gaussian_wave(dt, T["mu"], T["sigma"], T["amp"])
                )
        
        # Add baseline wander (slow sinusoidal variation)
        baseline = 0.02 * np.sin(2 * np.pi * 0.33 * time_point)
        
        # Add noise
        noise = np.random.normal(0, noise_level)
        
        ecg[i] = signal_value + baseline + noise
    
    # High-pass filter to remove DC drift (moving average subtraction)
    window_size = max(int(0.5 * sampling_rate), 1)
    if len(ecg) > window_size:
        moving_avg = np.convolve(ecg, np.ones(window_size) / window_size, mode='same')
        ecg = ecg - moving_avg
    else:
        # For very short signals, just subtract mean
        ecg = ecg - np.mean(ecg)
    
    # Normalize amplitude
    max_abs = np.max(np.abs(ecg))
    if max_abs > 1e-6:
        ecg = ecg / max_abs
    
    # Calculate actual heart rate from beat intervals
    if len(beat_starts) > 1:
        intervals = np.diff([b for b in beat_starts if 0 <= b <= duration])
        if len(intervals) > 0:
            avg_rr = np.mean(intervals)
            actual_hr = 60.0 / avg_rr if avg_rr > 0 else base_hr
        else:
            actual_hr = base_hr
    else:
        actual_hr = base_hr
    
    return ecg, actual_hr


# ============================================================================
# Mechanical Activity Generation
# ============================================================================

def generate_mechanical(
    ecg_signal: np.ndarray,
    scale: float = 1.2,
    delay_ms: float = 50.0,
    sampling_rate: int = 250
) -> np.ndarray:
    """
    Generate mechanical contraction signal based on ECG signal.
    
    The mechanical activity follows the electrical activity with a slight delay
    (electromechanical coupling delay). The contraction strength is proportional
    to the absolute value of the ECG signal.
    
    Parameters
    ----------
    ecg_signal : np.ndarray
        ECG signal array
    scale : float
        Scaling factor for mechanical amplitude (default: 1.2)
    delay_ms : float
        Delay between electrical and mechanical activity in milliseconds
        (default: 50 ms, typical electromechanical delay)
    sampling_rate : int
        Sampling rate in Hz (must match ECG sampling rate)
    
    Returns
    -------
    np.ndarray
        Mechanical contraction signal array
    """
    # Convert delay from milliseconds to samples
    delay_samples = int((delay_ms / 1000.0) * sampling_rate)
    
    # Shift ECG signal by delay (mechanical follows electrical)
    if delay_samples > 0:
        ecg_shifted = np.pad(ecg_signal, (delay_samples, 0), mode='constant')[:len(ecg_signal)]
    else:
        ecg_shifted = ecg_signal
    
    # Mechanical contraction is proportional to absolute value of ECG
    # This simulates how electrical depolarization triggers mechanical contraction
    mechanical = np.abs(ecg_shifted) * scale
    
    # Add smoothing to simulate muscle contraction dynamics (low-pass filter)
    # Simple moving average filter
    window_size = max(int(0.05 * sampling_rate), 1)  # 50ms smoothing window
    kernel = np.ones(window_size) / window_size
    mechanical = np.convolve(mechanical, kernel, mode='same')
    
    # Ensure non-negative (contraction is always positive)
    mechanical = np.maximum(mechanical, 0)
    
    # Normalize to 0-1 range for better visualization
    max_val = np.max(mechanical)
    if max_val > 1e-6:
        mechanical = mechanical / max_val
    
    return mechanical


# ============================================================================
# Main Simulation Function
# ============================================================================

def simulate_heart(
    condition: str = "normal",
    duration: float = 10.0,
    sampling_rate: int = 250,
    noise_level: float = 0.01,
    mechanical_scale: float = 1.2,
    mechanical_delay_ms: float = 50.0
) -> Dict:
    """
    Complete heart simulation: generates both ECG and mechanical signals.
    
    Parameters
    ----------
    condition : str
        Heart condition: "normal", "tachycardia", "bradycardia", "irregular"
    duration : float
        Duration of simulation in seconds
    sampling_rate : int
        Sampling rate in Hz
    noise_level : float
        ECG noise level
    mechanical_scale : float
        Scaling factor for mechanical signal
    mechanical_delay_ms : float
        Delay between electrical and mechanical activity (ms)
    
    Returns
    -------
    Dict
        Dictionary containing:
        - ecg_signal: ECG signal array
        - mechanical_signal: Mechanical contraction signal array
        - heart_rate: Calculated heart rate (bpm)
        - type: Heart condition type
        - time: Time array
        - sampling_rate: Sampling rate
    """
    # Generate ECG signal
    ecg_signal, heart_rate = generate_ecg(
        condition=condition,
        duration=duration,
        sampling_rate=sampling_rate,
        noise_level=noise_level
    )
    
    # Generate mechanical signal from ECG
    mechanical_signal = generate_mechanical(
        ecg_signal=ecg_signal,
        scale=mechanical_scale,
        delay_ms=mechanical_delay_ms,
        sampling_rate=sampling_rate
    )
    
    # Generate time array
    time = np.linspace(0, duration, len(ecg_signal))
    
    return {
        "ecg_signal": ecg_signal.tolist(),
        "mechanical_signal": mechanical_signal.tolist(),
        "heart_rate": round(heart_rate, 2),
        "type": condition.lower(),
        "time": time.tolist(),
        "sampling_rate": sampling_rate,
        "duration": duration
    }


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_signals(
    ecg: np.ndarray,
    mechanical: np.ndarray,
    time: Optional[np.ndarray] = None,
    condition: str = "normal",
    heart_rate: Optional[float] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot ECG and mechanical signals with comparison.
    
    Parameters
    ----------
    ecg : np.ndarray
        ECG signal array
    mechanical : np.ndarray
        Mechanical signal array
    time : np.ndarray, optional
        Time array (if None, generates from signal length)
    condition : str
        Heart condition label
    heart_rate : float, optional
        Heart rate to display
    save_path : str, optional
        Path to save the plot (if None, doesn't save)
    show_plot : bool
        Whether to display the plot (default: True)
    """
    # Generate time array if not provided
    if time is None:
        time = np.arange(len(ecg)) / 250.0  # Assume 250 Hz if not specified
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'Electro-Mechanical Heart Model - {condition.capitalize()} Rhythm' + 
                 (f' (HR: {heart_rate:.1f} bpm)' if heart_rate else ''), 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: ECG Signal
    axes[0].plot(time, ecg, 'b-', linewidth=1.5, label='ECG Signal')
    axes[0].set_ylabel('ECG Amplitude (mV)', fontsize=11)
    axes[0].set_title('Electrical Activity (ECG)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    axes[0].set_xlim([time[0], time[-1]])
    
    # Plot 2: Mechanical Signal
    axes[1].plot(time, mechanical, 'r-', linewidth=1.5, label='Mechanical Contraction')
    axes[1].set_ylabel('Contraction Strength (normalized)', fontsize=11)
    axes[1].set_title('Mechanical Activity (Contraction)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    axes[1].set_xlim([time[0], time[-1]])
    
    # Plot 3: Combined Comparison
    ax3_twin = axes[2].twinx()
    
    line1 = axes[2].plot(time, ecg, 'b-', linewidth=1.5, label='ECG Signal', alpha=0.7)
    line2 = ax3_twin.plot(time, mechanical, 'r-', linewidth=1.5, label='Mechanical Contraction', alpha=0.7)
    
    axes[2].set_xlabel('Time (seconds)', fontsize=11)
    axes[2].set_ylabel('ECG Amplitude (mV)', fontsize=11, color='b')
    ax3_twin.set_ylabel('Contraction Strength', fontsize=11, color='r')
    axes[2].set_title('Electrical vs Mechanical Activity Comparison', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([time[0], time[-1]])
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axes[2].legend(lines, labels, loc='upper right')
    
    axes[2].tick_params(axis='y', labelcolor='b')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


# ============================================================================
# Data Saving Functions
# ============================================================================

def save_signals_csv(
    data: Dict,
    filename: str = None
) -> str:
    """
    Save ECG and mechanical signals to CSV file.
    
    Parameters
    ----------
    data : Dict
        Dictionary from simulate_heart() containing signals
    filename : str, optional
        Output filename (if None, generates timestamped filename)
    
    Returns
    -------
    str
        Path to saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"heart_simulation_{data['type']}_{timestamp}.csv"
    
    # Prepare data for CSV
    time = data['time']
    ecg = data['ecg_signal']
    mechanical = data['mechanical_signal']
    
    # Write CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['time', 'ecg_signal', 'mechanical_signal'])
        
        # Write metadata as comments
        writer.writerow([f'# Condition: {data["type"]}'])
        writer.writerow([f'# Heart Rate: {data["heart_rate"]} bpm'])
        writer.writerow([f'# Duration: {data["duration"]} s'])
        writer.writerow([f'# Sampling Rate: {data["sampling_rate"]} Hz'])
        writer.writerow([])  # Empty row
        
        # Write data
        for t, e, m in zip(time, ecg, mechanical):
            writer.writerow([t, e, m])
    
    print(f"Data saved to CSV: {filename}")
    return filename


def save_signals_numpy(
    data: Dict,
    filename: str = None
) -> str:
    """
    Save ECG and mechanical signals to NumPy .npz file.
    
    Parameters
    ----------
    data : Dict
        Dictionary from simulate_heart() containing signals
    filename : str, optional
        Output filename (if None, generates timestamped filename)
    
    Returns
    -------
    str
        Path to saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"heart_simulation_{data['type']}_{timestamp}.npz"
    
    # Convert to numpy arrays
    np.savez(
        filename,
        ecg_signal=np.array(data['ecg_signal']),
        mechanical_signal=np.array(data['mechanical_signal']),
        time=np.array(data['time']),
        heart_rate=data['heart_rate'],
        condition=data['type'],
        duration=data['duration'],
        sampling_rate=data['sampling_rate']
    )
    
    print(f"Data saved to NumPy: {filename}")
    return filename


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Generate normal heart rhythm
    print("Generating normal heart rhythm...")
    result = simulate_heart(condition="normal", duration=10.0)
    
    print(f"Heart Rate: {result['heart_rate']} bpm")
    print(f"Signal Length: {len(result['ecg_signal'])} samples")
    
    # Plot signals
    plot_signals(
        np.array(result['ecg_signal']),
        np.array(result['mechanical_signal']),
        time=np.array(result['time']),
        condition=result['type'],
        heart_rate=result['heart_rate']
    )

