"""Minimal fsQCA example."""

import pandas as pd

from setqca import FSQCA, calibrate_direct

raw = pd.DataFrame(
    {
        "digital": [12, 24, 45, 60, 72, 88, 95, 35],
        "skills": [20, 35, 52, 64, 75, 82, 90, 44],
        "innovation": [15, 30, 48, 70, 78, 91, 96, 37],
    }
)

for column in raw.columns:
    raw[column] = calibrate_direct(raw[column], full_out=20, crossover=50, full_in=80)

model = FSQCA(consistency=0.8, pri=0.5, frequency=1)
result = model.fit(raw, outcome="innovation", conditions=["digital", "skills"])

print(result)
print(result.truth_table.to_frame())
