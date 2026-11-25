# Preparing a data file with per-point errors for eisyFIT

Use this when you want eisyFIT to derive the EEC fit's per-point weights
automatically from error estimates stored next to the impedance data
(**Fit → Edit point weights… → Import errors from data file**).

## File format

Any delimited text file that eisyFIT can import (CSV, tab-separated `.txt`,
or a BioLogic `.mpt` export) works. On import (**File → Import data…**,
`Ctrl+I`) set the delimiter (type `\t` for tab) and rows to skip; eisyFIT
recognizes common column names automatically, otherwise map them in the
import dialog.

Required impedance columns (one of the listed spellings is enough):

| Meaning | Recognized names |
| --- | --- |
| Frequency | `freq/Hz`, `frequency`, `f/Hz`, `freq_hz` |
| Re(Z) | `Re(Z)/Ohm`, `Re(Z)`, `Z'`, `real` |
| −Im(Z) | `-Im(Z)/Ohm`, `-Im(Z)`, `Z''`, `Im(Z)` (sign is handled) |

Optional: `cycle number` (repeat-loop label, enables multi-cycle files),
`time/s`, `<Ewe>/V` / `Ewe/V`, `<I>/mA` / `I/mA`.

## Error columns

Add **either** one modulus-error column **or** two component-error columns.
Include `err` in the column name so the dialog preselects it, e.g.:

| Mode | Columns | Example names |
| --- | --- | --- |
| Modulus | one column with σ\|Z\| (in ohm) | `err \|Z\|/Ohm` |
| Components | σRe(Z) and σ−Im(Z) (in ohm) | `err Re(Z)/Ohm`, `err -Im(Z)/Ohm` |

Rules:

- Values are **1-sigma errors σ, not variances**, in the same unit as Z (ohm).
- Every value must be **positive and finite**. Blank, zero, or negative
  entries are not importable: those points keep weight 1.0 and the dialog
  reports how many were skipped.
- In the components mode eisyFIT combines them as σ = √(σRe² + σIm²).
- Rows with blank/zero frequency are ignored, as in normal import.

## How eisyFIT uses them

Weights are `w = 1/σ` per point (equivalent to `curve_fit`'s `sigma`):

- modulus mode: `w = 1/σ|Z|`
- components mode: `w = 1/√(σRe² + σIm²)`

Points are matched to file rows **by frequency within the cycle** (rows may
be in any order; multi-cycle files are split on `cycle number`), so the error
columns just need to sit in the same rows as the corresponding spectrum
points. Applying the import composes with "Weight residuals by |Z|" as
`w/|Z|`. The resulting weights are stored on the cycle and saved in the
`.eisfit.json` project.

## Minimal example (CSV)

```csv
freq/Hz,Re(Z)/Ohm,-Im(Z)/Ohm,err |Z|/Ohm
100000,10.02,0.11,0.05
10000,10.85,1.20,0.06
1000,14.30,8.90,0.09
100,25.10,22.40,0.15
10,38.70,30.10,0.22
```

Or with component errors, add `err Re(Z)/Ohm,err -Im(Z)/Ohm` instead of the
modulus column.

## Workflow in eisyFIT

1. **File → Import data…** (`Ctrl+I`), choose the file, set delimiter/skip
   rows and column mapping if needed.
2. Optionally select several spectra in the explorer — the import then
   applies to **all selected spectra** (each cycle reads its own rows).
3. **Fit → Edit point weights…** → *Import errors from data file*. The
   default mode is **Re(Z) and −Im(Z) error columns**; switch to
   *Modulus |Z| error column* if your file has a single σ|Z| column.
4. Verify the preselected column names, click **Load errors as weights**,
   then **Apply**, and fit as usual. The status bar shows
   "custom weights on N points" and the fit diagnostics record the weighting.
