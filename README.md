# CLEAR

Experiment to calibrate and implement CLEAR pulses on a readout resonator using Qibocal/Qibolab.

## Contents

- `qibocal/calibration_exp.yml`: Qibocal calibration sequence.
- `script_py/parametri/risonatore.toml`: physical resonator parameters used by the CLEAR model.
- `script_py/amplitudes_CLEAR.ipynb`: notebook used to calculate the theoretical CLEAR amplitudes.
- `script_py/CLEAR_qibolab.py`: Qibolab sequence including CLEAR, Ramsey, and final measurement.
- `script_py/funz/`: Julia functions for the pulse model and simulation, called by `script_py/amplitudes_CLEAR.ipynb`.

## Experimental workflow

1. Run the Qibocal calibration.
2. Update `risonatore.toml` with `kappa`, `chi`, and `detuning`.
3. Run `amplitudes_CLEAR.ipynb`.
4. Rescale the CLEAR amplitudes with respect to the calibrated digital readout amplitude.
5. Insert the final amplitudes into `CLEAR_qibolab.py`.
6. Run the CLEAR + Ramsey sequence in Qibolab.
