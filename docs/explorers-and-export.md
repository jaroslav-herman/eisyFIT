# Explorers and export

The explorers help you compare spectra and fitted parameters across a whole project. The Export menu writes results to files.

## Spectra explorer

The **Spectra explorer** lists every loaded spectrum. Columns include:

- **Fitted** — whether an EEC fit is available,
- **DRT** — whether a DRT calculation is available,
- **EEC model** — the circuit string,
- **Source file**, **Cycle**, **Ecell_V**, **I_mA**, **time/s**,
- **Points** — number of points,
- **fmin_act_Hz** / **fmax_act_Hz** — active frequency limits.

Explorer features:

- Click column headers to sort; click again to reverse.
- Drag column headers to reorder columns.
- **Natural sort** uses natural number ordering for filenames and numeric text.
- **Restore** restores the saved column order.
- **Edit metadata…** adds or edits metadata columns for selected spectra, optionally deriving values from filenames with a regular expression.
- Select multiple rows with `Ctrl`/`Shift`, or select all with `Ctrl+A`.

Explorer action buttons:

- **Delete selected spectra** removes selected rows from the project.
- **Plot selected spectra** plots selected rows in the main plot.
- **Plot selected DRTs** plots DRTs for selected rows.
- **Plot cell/WE/CE** plots cell, working-electrode, and counter-electrode spectra together when available.
- **Fit parameters explorer** opens the EEC parameter explorer.
- **DRT parameters explorer** opens the DRT parameter explorer.

## Fit parameters explorer

The **Fit parameters explorer** plots EEC fit parameters against each other or against metadata such as cycle, voltage, current, or time.

Steps:

1. Select spectra in the Spectra Explorer.
2. Press **Fit parameters explorer**.
3. Choose X and Y axes.
4. Click points in the scatter plot to select them.
5. Optionally fit a curve: enable the curve-fit tool, type an equation with free parameters (for example `a*x + b` or `a*exp(-b*x) + c`), pick points, and solve for the parameters.

## DRT parameters explorer

The **DRT parameters explorer** does the same for DRT peak parameters and DRT metadata. It is useful for spotting trends in peak area, peak tau, or peak width across a series.

## Export menu

| Export | Output |
| --- | --- |
| Export fit parameters - all spectra… | CSV with fitted EEC parameters and spectrum metadata for all spectra |
| Export fit parameters - selected spectra… | CSV for the selected spectra |
| Export to Python - all spectra… | Fit CSV files plus a ready-to-run pandas script, opened in an available editor |
| Export to Python - selected spectra… | Same for the selected spectra |
| Export DRTs - all spectra… | Saved DRT points for all spectra |
| Export DRTs - selected spectra… | Saved DRT points for the selected spectra |

## Plot context menu

Right-click a plot for:

- **Save graph** — save the rendered plot as an image,
- **Copy to Clipboard** — copy the rendered plot image,
- **Export data** — export the displayed plot series to CSV,
- **Reset view** — reset zoom and pan.

## Practical workflow

```text
Fit or analyze a series → open Fit/DRT parameters explorer → plot parameter vs cycle/time/voltage
→ fit a trend curve → export fit parameters or DRTs
```
