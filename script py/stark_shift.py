import json
import socket
import numpy as np
from tqdm import tqdm

from qibosoq.client import execute
from qibosoq.components.base import Qubit, OperationCode, Config, Sweeper, Parameter
from qibosoq.components.pulses import Rectangular, Measurement

filename = "data/stark_shift_data_0.npz"

HOST = "jarvis.mib.infn.it"
PORT = 6000

DAC_PROBE = 2
DAC_DRIVE = 1
ADC_RO = 0

RO_FREQ = 7269.425000  # MHz
RO_AMP = np.linspace(0.0025, 0.02, 25)
# PROBE_DURATION = 2.5  # us
RO_DURATION = 4

DRIVE_FREQ = np.linspace(5060-50, 5260-50, 200)  # MHZ
DRIVE_AMP = 0.01
DRIVE_DURATION = 1

PROBE_DURATION = 1 + DRIVE_DURATION + RO_DURATION

results = []

for ro_amp in tqdm(RO_AMP, desc="Outer"):
    for drive_freq in tqdm(DRIVE_FREQ, desc="Inner", leave=False):
        pulse_1 = Rectangular(
            frequency=RO_FREQ,
            amplitude=ro_amp,
            relative_phase=0,
            start_delay=0,
            duration=DRIVE_DURATION * 2,
            name="readout_probe",
            type="drive",
            dac=DAC_PROBE,
            adc=ADC_RO,
        )

        pulse_2 = Rectangular(
            frequency=drive_freq,
            amplitude=DRIVE_AMP,
            relative_phase=0,
            start_delay=DRIVE_DURATION,
            duration=DRIVE_DURATION,
            name="drive_pulse",
            type="drive",
            dac=DAC_DRIVE,
            adc=ADC_RO,
        )

        meas = Rectangular(
            name="ro_pulse",
            amplitude=0.02,
            relative_phase=0,
            frequency=RO_FREQ,
            start_delay=DRIVE_DURATION,
            duration=RO_DURATION,
            type="readout",
            dac=DAC_PROBE,
            adc=ADC_RO,
        )

        sequence = [pulse_1, pulse_2, meas]

        config = Config(relaxation_time=0, reps=8000)
        qubit = Qubit()

        server_commands = {
            "operation_code": OperationCode.EXECUTE_PULSE_SEQUENCE,
            "cfg": config,
            "sequence": sequence,
            "qubits": [qubit],
        }

        i, q = execute(server_commands, HOST, PORT)

        results.append(
            {
                "probe_amplitude": ro_amp,
                "drive_frequency": drive_freq,
                "I": np.mean(i),
                "Q": np.mean(q),
            }
        )


# Conversione in array strutturato
data = {
    "probe_amplitude": np.array([r["probe_amplitude"] for r in results]),
    "drive_frequency": np.array([r["drive_frequency"] for r in results]),
    "I": np.array([r["I"] for r in results]),
    "Q": np.array([r["Q"] for r in results]),
}


# Salvataggio
np.savez(filename, **data)

print("Dataset salvato: stark_shift_data.npz")

import matplotlib.pyplot as plt

data = np.load(filename)
I = data["I"]
Q = data["Q"]


probe_amp = data["probe_amplitude"]
drive_freq = data["drive_frequency"]

# Ricostruzione griglia
amp_values = np.unique(probe_amp)
freq_values = np.unique(drive_freq)

I_map = I.reshape(len(amp_values), len(freq_values))
Q_map = Q.reshape(len(amp_values), len(freq_values))

# Magnitudine IQ
mag_map = np.sqrt(I_map**2 + Q_map**2)

# Plot
plt.figure(figsize=(8, 5))

plt.pcolormesh(freq_values, amp_values, mag_map, shading="auto")

plt.xlabel("Drive frequency (MHz)")
plt.ylabel("Probe amplitude")
plt.colorbar(label="|IQ|")

plt.title("Stark shift map")
plt.tight_layout()
plt.savefig("stark.png")
plt.show()
