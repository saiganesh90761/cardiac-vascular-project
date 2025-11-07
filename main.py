"""
Main Example Script for Electro-Mechanical Heart Simulator

This script demonstrates the complete heart simulation system by generating
and visualizing signals for all supported heart conditions.

Run this script to see examples of:
- Normal heart rhythm
- Tachycardia (fast heart rate)
- Bradycardia (slow heart rate)
- Irregular heart rhythm

All signals are plotted and saved to files for ML model training.
"""

import numpy as np
from heart_simulator import (
    simulate_heart,
    plot_signals,
    save_signals_csv,
    save_signals_numpy
)


def run_simulation_for_condition(
    condition: str,
    duration: float = 10.0,
    save_data: bool = True,
    show_plots: bool = True
):
    """
    Run complete simulation for a specific heart condition.
    
    Parameters
    ----------
    condition : str
        Heart condition: "normal", "tachycardia", "bradycardia", "irregular"
    duration : float
        Duration of simulation in seconds
    save_data : bool
        Whether to save data to files
    show_plots : bool
        Whether to display plots
    """
    print(f"\n{'='*60}")
    print(f"Simulating {condition.upper()} Heart Rhythm")
    print(f"{'='*60}")
    
    # Generate heart signals
    result = simulate_heart(
        condition=condition,
        duration=duration,
        sampling_rate=250,
        noise_level=0.01,
        mechanical_scale=1.2,
        mechanical_delay_ms=50.0
    )
    
    # Display information
    print(f"\nCondition: {result['type']}")
    print(f"Heart Rate: {result['heart_rate']} bpm")
    print(f"Duration: {result['duration']} s")
    print(f"Sampling Rate: {result['sampling_rate']} Hz")
    print(f"Signal Length: {len(result['ecg_signal'])} samples")
    
    # Convert to numpy arrays for plotting
    ecg = np.array(result['ecg_signal'])
    mechanical = np.array(result['mechanical_signal'])
    time = np.array(result['time'])
    
    # Plot signals
    if show_plots:
        plot_signals(
            ecg=ecg,
            mechanical=mechanical,
            time=time,
            condition=result['type'],
            heart_rate=result['heart_rate'],
            save_path=f"heart_plot_{condition}.png",
            show_plot=True
        )
    
    # Save data
    if save_data:
        csv_file = save_signals_csv(result, f"heart_data_{condition}.csv")
        npz_file = save_signals_numpy(result, f"heart_data_{condition}.npz")
        print(f"\nSaved files:")
        print(f"  - CSV: {csv_file}")
        print(f"  - NumPy: {npz_file}")
    
    return result


def main():
    """
    Main function to run simulations for all heart conditions.
    """
    print("\n" + "="*60)
    print("ELECTRO-MECHANICAL HEART SIMULATOR")
    print("="*60)
    print("\nThis simulator generates:")
    print("  1. ECG signals (electrical activity)")
    print("  2. Mechanical contraction signals")
    print("  3. Visualizations and data files")
    print("\nSupported conditions:")
    print("  - normal: Normal heart rhythm (60-100 bpm)")
    print("  - tachycardia: Fast heart rate (>100 bpm)")
    print("  - bradycardia: Slow heart rate (<60 bpm)")
    print("  - irregular: Irregular heart rhythm")
    print("\n" + "="*60)
    
    # Configuration
    DURATION = 10.0  # seconds
    SAVE_DATA = True
    SHOW_PLOTS = True
    
    # Run simulations for all conditions
    conditions = ["normal", "tachycardia", "bradycardia", "irregular"]
    
    results = {}
    
    for condition in conditions:
        try:
            result = run_simulation_for_condition(
                condition=condition,
                duration=DURATION,
                save_data=SAVE_DATA,
                show_plots=SHOW_PLOTS
            )
            results[condition] = result
        except Exception as e:
            print(f"\nError simulating {condition}: {e}")
            continue
    
    # Summary
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    
    for condition, result in results.items():
        print(f"\n{condition.upper()}:")
        print(f"  Heart Rate: {result['heart_rate']} bpm")
        print(f"  ECG Signal Range: [{min(result['ecg_signal']):.3f}, {max(result['ecg_signal']):.3f}]")
        print(f"  Mechanical Signal Range: [{min(result['mechanical_signal']):.3f}, {max(result['mechanical_signal']):.3f}]")
    
    print("\n" + "="*60)
    print("All simulations completed successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - CSV files: heart_data_*.csv (for ML model training)")
    print("  - NumPy files: heart_data_*.npz (for ML model training)")
    print("  - Plot images: heart_plot_*.png (visualizations)")
    print("\nYou can now use these files with your ML model!")


if __name__ == "__main__":
    main()

