Split: **random** | patches: 27000 (train 18900 / val 4050 / test 4050)

| Model | Test accuracy | Macro-F1 | CV accuracy |
|---|---|---|---|
| Random Forest (13 SAR features) | 0.735 | 0.723 | 0.730 +/- 0.005 (StratifiedKFold(5)) |
| Keras CNN (VV, VH, ratio) | 0.858 | 0.852 | - |

Polarisation ablation (RF test accuracy):

| Features | Accuracy |
|---|---|
| VV only | 0.615 |
| VH only | 0.605 |
| VV + VH | 0.714 |
| all features | 0.735 |