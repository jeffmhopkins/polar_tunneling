# Lift robustness (log-loss gain x 1000 of +tunnel over +prev, test Aug-Oct)

|   commit ms | pairs                | outcome       |   gain x1000 |   null mean |   null max |   null draws | share of nulls beaten   |
|------------:|:---------------------|:--------------|-------------:|------------:|-----------:|-------------:|:------------------------|
|         125 | all pairs            | whiff | swing |        -0.58 |       -0.86 |      -0.57 |            3 | 67%                     |
|         125 | all pairs            | swing         |        -0.06 |       -0.75 |      -0.65 |            3 | 100%                    |
|         125 | different-type pairs | whiff | swing |         0.02 |       -0.61 |      -0.54 |            3 | 100%                    |
|         125 | different-type pairs | swing         |         0.51 |       -0.69 |      -0.56 |            3 | 100%                    |
|         175 | all pairs            | whiff | swing |        -0.54 |       -0.69 |      -0.3  |           10 | 70%                     |
|         175 | all pairs            | swing         |        -0.41 |       -0.89 |      -0.67 |           10 | 100%                    |
|         175 | different-type pairs | whiff | swing |        -0.14 |       -0.34 |       0.11 |            3 | 67%                     |
|         175 | different-type pairs | swing         |         0.19 |       -0.6  |      -0.34 |            3 | 100%                    |
|         225 | all pairs            | whiff | swing |        -0.61 |       -0.78 |      -0.71 |            3 | 100%                    |
|         225 | all pairs            | swing         |        -0.11 |       -0.69 |      -0.58 |            3 | 100%                    |
|         225 | different-type pairs | whiff | swing |        -0.08 |       -0.67 |      -0.46 |            3 | 100%                    |
|         225 | different-type pairs | swing         |         0.02 |       -0.47 |      -0.28 |            3 | 100%                    |
