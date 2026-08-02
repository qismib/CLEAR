# CLEAR

Esperimento per calibrare e implementare impulsi CLEAR su un risonatore di readout con Qibocal/Qibolab.

## Contenuto

- `qibocal/calibration_exp.yml`: sequenza di calibrazione Qibocal.
- `script py/parametri/risonatore.toml`: parametri fisici del risonatore usati dal modello CLEAR.
- `script py/amplitudes_CLEAR.ipynb`: notebook per calcolare le ampiezze teoriche CLEAR.
- `script py/CLEAR_qibolab.py`: sequenza Qibolab CLEAR + Ramsey + misura finale.
- `script py/funz/`: funzioni Julia per il modello e la simulazione dell'impulso, chiamae in `script py/amplitudes_CLEAR.ipynb`.

## Flusso sperimentale

1. Eseguire la calibrazione Qibocal.
2. Aggiornare `risonatore.toml` con `kappa`, `chi` e `detuning`.
3. Rilanciare `amplitudes_CLEAR.ipynb`.
4. Riscalare le ampiezze CLEAR rispetto all'ampiezza digitale di readout calibrata.
5. Inserire le ampiezze finali in `CLEAR_qibolab.py`.
6. Eseguire la sequenza CLEAR + Ramsey in Qibolab.
