import numpy as np
import matplotlib.pyplot as plt
FOLDER = "data/"
data = np.load(FOLDER + "stark_shift_data_0.npz")
chi_MHz = (7269.555-7267.185)/2  # chi = (fr(1) - fr(0))/2

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
plt.savefig(FOLDER + "stark_map.png", bbox_inches="tight")

plt.show()
f_res = []

for row in mag_map:
    idx = np.argmax(row)
    f_res.append(freq_values[idx])

f_res = np.array(f_res)


plt.figure(figsize=(6, 4))
plt.plot(amp_values, f_res, "o-")

plt.xlabel("Probe amplitude")
plt.ylabel("Resonance frequency (MHz)")
plt.title("Stark shifted resonance")

plt.grid()
plt.savefig(FOLDER + "freq_vs_amp.png", bbox_inches="tight")

plt.show()


f0 = f_res[0]  # MHz, assumendo primo punto quasi vuoto
delta_f = f_res - f0  # MHz

n_photons = -delta_f / (2 * chi_MHz)

plt.figure(figsize=(6, 4))

plt.plot(amp_values**2, n_photons, "o-")

k, q = np.polyfit(amp_values**2, n_photons, 1)
plt.plot(amp_values**2, k * (amp_values**2) + q, "--", label=f"fit lineare: k={k:.4g}")
plt.legend()
print(f"Coefficiente angolare k = {k}")

plt.xlabel("Probe amplitude")
plt.ylabel("Mean photon number $\\bar{n}$")

plt.title("Resonator photon population")

plt.grid()
plt.savefig(FOLDER + "nphotons_vs_amp.png", bbox_inches="tight")

plt.show()

# Salvataggio del coefficiente angolare k (e dell'intercetta q) del fit lineare
# n_photons = k * amp^2 + q, cosi' da poterlo riusare in amplitudes_CLEAR.ipynb
# senza ricopiarlo a mano.
import json

fit_results = {"k": float(k), "q": float(q)}
fit_path = FOLDER + "stark_fit_results.json"
with open(fit_path, "w") as f:
    json.dump(fit_results, f, indent=2)

print(f"Fit salvato in {fit_path}: k = {k}, q = {q}")


