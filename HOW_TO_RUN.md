# Quick Start Guide - Electro-Mechanical Heart Simulator

## Step 1: Install Dependencies

First, make sure you have all required packages installed:

```powershell
pip install -r requirements.txt
```

Or install individually:
```powershell
pip install numpy matplotlib scipy
```

## Step 2: Run the Simulator

### Option A: Run the Complete Example (All Conditions)
```powershell
python main.py
```

This will:
- Generate ECG and mechanical signals for all 4 conditions (normal, tachycardia, bradycardia, irregular)
- Display plots for each condition
- Save CSV and NumPy files for ML training
- Show summary statistics

### Option B: Use the Simulator in Your Own Code

```python
from heart_simulator import simulate_heart, plot_signals
import numpy as np

# Generate signals for a specific condition
result = simulate_heart(
    condition="normal",      # or "tachycardia", "bradycardia", "irregular"
    duration=10.0,           # seconds (5-20 recommended)
    sampling_rate=250,       # Hz
    noise_level=0.01
)

# Access the results
print(f"Heart Rate: {result['heart_rate']} bpm")
print(f"ECG Signal Length: {len(result['ecg_signal'])} samples")

# Plot the signals
plot_signals(
    np.array(result['ecg_signal']),
    np.array(result['mechanical_signal']),
    time=np.array(result['time']),
    condition=result['type'],
    heart_rate=result['heart_rate']
)

# Save data
from heart_simulator import save_signals_csv, save_signals_numpy
save_signals_csv(result, "my_heart_data.csv")
save_signals_numpy(result, "my_heart_data.npz")
```

## Step 3: View Generated Files

After running, you'll find:
- `heart_data_*.csv` - CSV files for ML training
- `heart_data_*.npz` - NumPy files for ML training  
- `heart_plot_*.png` - Visualization plots

## Troubleshooting

If you get import errors:
```powershell
# Make sure you're in the project directory
cd C:\Users\embot\OneDrive\Desktop\Cardiac

# Verify Python can find the module
python -c "import heart_simulator; print('OK')"
```

If matplotlib doesn't display plots:
- On Windows, you may need: `pip install tkinter` (usually comes with Python)
- Or use: `plot_signals(..., show_plot=False)` and check saved PNG files


