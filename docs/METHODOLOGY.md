# Methodology Notes

## Target Variable

`Emission_Intensity` is defined as:

```
Emission_Intensity = Cogen_Adjusted_Emission / Production
```

measured in tonnes CO2e per cubic meter of production.

## Avoiding Target Leakage

An earlier iteration of this project included `Cogen_Adjusted_Emission` and a
derived ratio (`Emission / Production`) as **input features**. Because that
ratio is mathematically identical to the target variable, models trained on
it achieved a spurious R² of ~1.00 — they were not predicting anything, just
reproducing the definition of the label.

**This repository's pipeline (`src/data_processing.py`) deliberately excludes
any feature derived from `Cogen_Adjusted_Emission`.** Only information that
would plausibly be available *before* observing a facility's emissions is
used:

| Included (legitimate) | Excluded (leakage) |
|---|---|
| Production, log(Production) | Cogen_Adjusted_Emission |
| Year, facility age, time trend | log(Emission) |
| Subsector, extraction technology, product (encoded) | Emission / Production ratio |
| Facility historical average production | any transform of the above |
| Years facility has been operating | |
| Technology × Product, Subsector × Year interactions | |

## Resulting Performance

With leakage removed, the best model (Random Forest) achieves:

- **Test R² ≈ 0.50** — explains about half the variance in emissions
  intensity using only production, technology, and temporal information.
- **Test RMSE ≈ 0.74 tCO2e/m³**

This is a realistic, defensible predictive result. It is lower than the
inflated ~1.00 R² from the leaky version, but it reflects genuine predictive
signal rather than circular calculation, and it is the number that should be
reported and interpreted.

## Feature Importance (Leakage-Free)

| Feature | Random Forest Importance |
|---|---|
| Log(Production) | ~35% |
| Production | ~26% |
| Facility average production | ~18% |
| Extraction technology | ~4% |
| Technology × Product interaction | ~3% |

**Takeaway:** production scale and operational history are the dominant
drivers of emissions intensity that can be captured with the available
features; extraction technology and temporal effects contribute secondary,
smaller signal. Roughly half the variance remains unexplained, pointing to
the need for additional facility-level operational data (equipment age,
maintenance, energy source mix, carbon capture deployment) in future work.

## General Lesson

Watch for these red flags of target leakage in any regression project:

1. Near-perfect R² (> 0.99) on held-out data.
2. A single feature dominating importance (> 90%).
3. A feature that is a direct arithmetic transform of the target.
4. Inputs that would not actually be known at prediction time.

When in doubt, ask: *"Would I have this information before I know the
answer I'm trying to predict?"* If not, the feature does not belong in the
model.
