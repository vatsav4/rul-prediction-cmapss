# Dataset

The NASA **C-MAPSS Turbofan Engine Degradation Simulation** dataset is **not
committed to this repository**. Download it and place the files here.

## Where to get it

NASA Prognostics Center of Excellence (PCoE) Data Set Repository —
*"Turbofan Engine Degradation Simulation Data Set"*:
<https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>

## Expected layout

```
data/cmapss/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
├── train_FD002.txt ... (FD002-FD004, for later work)
└── readme.txt
```

## File format

Space-separated, no header. Column order is fixed:

| Position | Field |
|---|---|
| 1 | `unit` — engine number |
| 2 | `cycle` — time index for that engine |
| 3–5 | `op_setting_1..3` — operational settings |
| 6–26 | `sensor_1..21` — sensor measurements |

In the training files each engine runs to failure, so its last cycle is the
point of failure. In the test files each engine's record is truncated early,
and `RUL_FDxxx.txt` holds the true remaining life at that cut-off — one value
per engine, in engine order.

## Citation

A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage propagation modeling for
aircraft engine run-to-failure simulation," *International Conference on
Prognostics and Health Management (PHM)*, Denver, CO, USA, 2008, pp. 1–9.
