# Deep Learning-Based Remaining Useful Life Prediction for Industrial Rotating Equipment

An LSTM framework for **Remaining Useful Life (RUL)** prediction, validated on the NASA
C-MAPSS turbofan degradation benchmark and intended for application to rotating equipment
on a commercial-vehicle chassis assembly line.

> M.Tech (Artificial Intelligence & Machine Learning) dissertation — BITS Pilani, WILP
> Course **AITMZG628T** · Student **Komatlapalli Vatsav Sai Ram** (2024TM05021)
> Dissertation work carried out at Tata Motors Limited, Dharwad
> Supervisor: Maxim Quadras, Deputy General Manager

---

## Why this project

Maintenance on the chassis assembly line is largely preventive (time-based) or reactive.
Under either regime a component is replaced earlier than necessary, or only after a failure
has already stopped production. Condition-based maintenance fixes this by acting on the
measured health of the equipment — and its central requirement is an estimate of how much
service life an asset has left.

Plant equipment rarely fails, and a single semester cannot observe an asset through a
complete run-to-failure cycle, yet supervised RUL models need run-to-failure trajectories in
order to learn. The work is therefore split in two:

| | Phase 1 — benchmark (this repository) | Phase 2 — plant application |
|---|---|---|
| Data | NASA C-MAPSS, 100 engines run to failure | Chassis line conveyor drive |
| Labels | Yes — exact cycles remaining | None; the asset has not failed |
| Task | Supervised RUL regression | Unsupervised health index |
| Output | A RUL value in cycles | A degradation trend |
| Scored by | RMSE and the PHM scoring function | Trend behaviour vs maintenance records |

Phase 1 proves the method works, measurably and comparably against published results.
Phase 2 applies the same methodology — not the trained weights — to a real asset.

---

## Results (C-MAPSS FD001)

| Model | RMSE (cycles) | PHM score |
|---|---|---|
| XGBoost | **12.82** | **261.45** |
| LSTM | 13.48 | 294.70 |
| Random Forest | 16.24 | 631.25 |

Lower is better for both. **RMSE** is the symmetric average error in cycles. The **PHM score**
is the original challenge metric and is deliberately asymmetric — a *late* prediction (the
model claims more remaining life than the asset has) is penalised far more heavily than an
early one, which matches the maintenance reality that a missed failure costs more than a
needless inspection.

The LSTM improves on Random Forest decisively on both metrics. XGBoost remains marginally
ahead, and that is reported as found rather than tuned away. FD001 is the simplest subset —
one operating condition, one fault mode — which is precisely the case where sequence
modelling has least to contribute, because the mapping from sensor values to remaining life
is close to stationary and a strong tree ensemble on flattened windows can capture it
directly. The recurrent model's advantage is expected to widen on FD002/FD004, where six
operating conditions make the same sensor reading mean different things depending on context.
Testing that is part of the remaining work.

### A methodological finding: normalise the target, not just the inputs

Input normalisation is treated as routine in the literature. Normalising the **regression
target** turned out to decide whether the network trained at all.

| Target handling | RMSE | PHM score | Prediction spread |
|---|---|---|---|
| Raw target (0–125) | 40.38 | 16,340 | constant 79.2 (std 0.0) |
| Scaled target (0–1) | **13.48** | **294.70** | 9.9 – 125.1 (std 39.4) |

Trained on the raw 0–125 target, the initial MSE is roughly the variance of the labels
(~1,750). The optimiser settles at the label mean and the network emits a single constant —
79.2 cycles — for every engine. Validation loss then plateaus there, so early stopping ends
training and preserves the failure behind a converged-looking loss curve. Scaling the target
to [0, 1] and rescaling predictions afterwards resolves it completely, with no change to the
architecture or the data.

Two safeguards are now permanent in `04`: the score of a trivial constant predictor is
computed *before* training as a reference point, and an automatic check flags any model whose
predictions show negligible variance.

---

## Method

**Data.** C-MAPSS FD001 — 100 engines run to failure (20,631 cycles) for training, 100
truncated engines for test, 21 sensors plus 3 operational settings per cycle. Engine lives
range from 128 to 362 cycles, so elapsed age alone is a poor predictor; the model has to read
the sensors.

**Preprocessing.** Seven constant or near-constant sensors are dropped, leaving 14 informative
channels. Labels use the standard **piecewise-linear RUL** target capped at 125 cycles — a
healthy engine's true remaining life is not recoverable from flat sensors, so an uncapped
linear target asks the model to fit noise. Sensors are min-max scaled with the scaler **fitted
on training data only** to avoid leakage. Each engine's history is cut into overlapping
30-cycle windows, giving 17,731 training sequences of shape 30 × 14. Each test engine
contributes a single window — its last 30 cycles — producing one prediction per engine, as the
benchmark protocol requires.

**Models.** Random Forest and XGBoost on flattened 420-feature windows as baselines, and a
stacked LSTM (64 → 32 units, dropout 0.2, dense 16 → 1; 33,185 parameters) that reads the
window as a sequence. Flattening discards the time ordering, which is exactly the structure
the recurrent model is meant to exploit. The network is deliberately small because FD001 has
only 100 engines and its windows overlap heavily.

**Training.** Adam (lr 1e-3), MSE loss, batch 256, 15% validation split, early stopping with
patience 25 restoring best weights, learning rate halved on plateau. Converged in 103 epochs.

---

## Repository layout

```
.
├── notebooks/
│   ├── 01_cmapss_fd001_eda.ipynb            # load, explore, select sensors (outputs included)
│   ├── 02_cmapss_fd001_preprocessing.ipynb  # RUL labelling, scaling, windowing
│   ├── 03_cmapss_fd001_baselines.ipynb      # Random Forest + XGBoost, metrics defined
│   └── 04_cmapss_fd001_lstm.ipynb           # LSTM, training, evaluation, comparison
├── data/README.md                           # how to obtain C-MAPSS (not committed)
├── results/model_comparison_FD001.csv
├── requirements.txt
└── LICENSE
```

Run the notebooks in order — each writes artefacts the next one reads.

---

## Running it

**Google Colab (recommended).** Open a notebook in Colab, set a GPU runtime for `04`
(*Runtime → Change runtime type → T4 GPU*), and place the C-MAPSS files in
`MyDrive/RUL_Dissertation/data/cmapss/`. The setup cell mounts Drive automatically.

**Locally.**

```bash
git clone https://github.com/<your-username>/rul-prediction-cmapss.git
cd rul-prediction-cmapss
pip install -r requirements.txt
# put the C-MAPSS .txt files in data/cmapss/  (see data/README.md)
jupyter notebook
```

The setup cell detects that it is not running in Colab and falls back to the repository
folder, so no path edits are needed.

---

## Reproducibility and known limitations

Seeds are fixed (`SEED = 42`) across NumPy, TensorFlow and scikit-learn, and every
intermediate artefact — the fitted scaler, the windowed arrays, the trained model — is saved.
Three limitations are worth stating plainly:

The 15% validation split shuffles **windows** rather than engines, so overlapping windows from
the same engine can fall on both sides of the split. This makes validation loss a slightly
optimistic estimate of generalisation. It does not affect the reported test results, which use
100 entirely separate engines, but splitting by engine is the cleaner design and is on the
list to fix.

The RUL cap of 125 and the window length of 30 are adopted from the literature for
comparability, not tuned on this data.

Results come from a single run with a fixed seed. Reporting a mean and standard deviation over
several seeds would be a stronger claim.

---

## Roadmap

- **SHAP explainability** — identify which sensors and which positions within the window drive
  the RUL estimate, since a maintenance engineer cannot act on an unexplained number.
- **FD002 – FD004** — test whether the LSTM's advantage widens as operating conditions and
  fault modes multiply.
- **Plant application** — an unsupervised health index on chassis line conveyor drive data
  (movement duration, breakaway current, mean running current, energy per move), framed as
  early degradation-trend detection rather than end-of-life prediction.

---

## References

1. A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage propagation modeling for aircraft engine run-to-failure simulation," *PHM*, Denver, CO, USA, 2008, pp. 1–9.
2. G. S. Babu, P. Zhao and X.-L. Li, "Deep convolutional neural network based regression approach for estimation of remaining useful life," *DASFAA*, LNCS 9642, Springer, 2016, pp. 214–228.
3. S. Zheng, K. Ristovski, A. Farahat and C. Gupta, "Long short-term memory network for remaining useful life estimation," *IEEE ICPHM*, Dallas, TX, USA, 2017, pp. 88–95.
4. X. Li, Q. Ding and J.-Q. Sun, "Remaining useful life estimation in prognostics using deep convolution neural networks," *Reliability Engineering & System Safety*, vol. 172, pp. 1–11, 2018.
5. Y. Wu, M. Yuan, S. Dong, L. Lin and Y. Liu, "Remaining useful life estimation of engineered systems using vanilla LSTM neural networks," *Neurocomputing*, vol. 275, pp. 167–179, 2018.
6. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," *NeurIPS*, vol. 30, 2017, pp. 4765–4774.

---

## Licence

Code is released under the MIT Licence (see `LICENSE`). The NASA C-MAPSS dataset is not
covered by that licence and is not redistributed here.
