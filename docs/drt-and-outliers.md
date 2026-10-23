# DRT and outliers

eisyFIT uses DRT (distribution of relaxation times) both as an analysis view and as a helper for cleaning spectra and initializing EEC fits.

## Ridge DRT and Hybrid DRT

Choose the method in the **DRT analysis** group:

- **Ridge DRT** — hierarchical Bayesian ridge inversion based on `bayes-drt2`.
- **Hybrid DRT** — probabilistic DRT analysis with a dual regression-classification framework.

Useful entry points:

- `Alt+R` calculates DRT for the current spectrum.
- `Alt+C` switches between Ridge DRT and Hybrid DRT.
- In **DRT** analysis mode, **Calculate DRT** computes DRT for selected spectra.
- **Auto model: selected (Hybrid DRT)** uses a Hybrid DRT analysis to suggest an EEC model for selected spectra.

## Viewing DRT results

Plot controls:

- **Show DRT** shows or hides the DRT panel.
- **Show DRT fit** shows the fitted DRT peak model.
- **Show DRT recovered** shows the spectrum recovered from the DRT model.
- **Show KK residuals** shows Kramers–Kronig residuals.

The explorer **DRT** column marks spectra with an available DRT calculation.

## Outlier detection

Outlier detection never runs automatically.

### Ridge-analysis outliers

In the **Selection** group:

1. Set **Outlier threshold**.
2. Press **Outliers**.

For the current spectrum, eisyFIT:

1. runs bayes-drt2 ridge analysis on active points only,
2. applies the resulting outlier mask without reactivating excluded points,
3. initializes the EEC fit from the ridge ohmic resistance, inductance, and strongest DRT peaks.

The same action is available as **Initial values** in the **Actions** group and as `Alt+V`.

### Deterministic outliers

1. Set **Deterministic threshold**.
2. Press **Remove deterministic outliers**.

This removes deterministic outliers from active points.

### Manual point editing

1. Press **Edit points: Off** until it reads **Edit points: On**.
2. Click points in the Nyquist/Bode plot to toggle include/exclude.
3. Enable **Show all points** to see excluded points.
4. Press **Active points** to fit the view to active points.
5. Press **Reset points** to include all points and clear detected outliers.

## DRT peak model

In **DRT** analysis mode:

- Add peaks with **Add Gaussian peak**, **Add Lorentzian peak**, **Add Voigt peak**, or **Add HN peak**.
- Edit peak parameters in the peak table.
- Press **Fit peaks** to fit the DRT model to the measured spectrum.
- Press **Send initials** to send DRT-derived initial values to the EEC circuit.
- Press **Remove all peaks** to clear the peak model.
- **Open DRT analysis window** opens a detached peak editor.

Peak-parameter values can be copied to selected spectra with **Apply all to selected**, or field-by-field with **Apply Fix**, **Apply Initial**, **Apply Lower**, and **Apply Upper**.

## Practical workflow

```text
Set frequency window → Initial values / Outliers → review excluded points
→ Fit spectrum → Show DRT / Show KK residuals → refine or export
```

## Interpretation notes

- A point is active through the cycle's included mask, which intersects manual inclusion with the frequency window.
- Outlier actions clear manual inclusion for rejected points, so previously excluded points stay excluded.
- DRT peaks are a model of the spectrum; use the recovered spectrum and residuals to judge whether the peak model is adequate.
- Kramers–Kronig residuals are a linearity/causality/stability check, not a replacement for physical interpretation.
