# eisyFIT overview

[eisyFIT](https://github.com/jaroslav-herman/eisyFIT) is a desktop application for exploring, cleaning, and fitting electrochemical impedance spectroscopy (EIS) data exported by BioLogic as `.mpt` (PEIS) files.

The main window combines:

- an interactive Nyquist/Bode plot with point editing, zoom, pan, and visibility toggles,
- an editable equivalent-circuit parameter table,
- cycle-aware spectrum state,
- DRT-assisted outlier detection and EEC initialization,
- batch fitting and saved procedures,
- a sortable spectra explorer with metadata columns,
- DRT and EEC fit-parameter explorers with an interactive curve-fit tool,
- exports of fitted parameters, DRTs, plot data, and ready-to-run Python workspaces.

## Analysis modes

Use the **Analysis mode** selector at the top of the right-hand control panel:

- **EEC** — fit the displayed spectrum (or selected spectra) with an equivalent-circuit model. Shows the fitting model controls, circuit parameter table, selection/outlier controls, and EEC actions.
- **DRT** — analyze selected spectra with Ridge DRT or Hybrid DRT. Shows peak controls for building a DRT model, fitting it, and sending its initials to the EEC circuit.
- **Spectra Simulator** — generate a synthetic spectrum from an EEC model and optional noise, then add it to the Spectra Explorer.

## Window layout

| Area | Purpose |
| --- | --- |
| Plot area | Nyquist or Bode view of active spectra, optional DRT panel, KK residuals, EEC fit, and DRT fit overlays |
| Plot controls | Edit points, auto-fit points, reset view, show/hide points and overlays, switch Nyquist/Bode |
| Spectra explorer | All loaded spectra with fitted/DRT markers, model, source file, cycle, voltage, current, time, point count, and active frequency limits |
| Control panel | Analysis mode, fitting model, circuit parameters, frequency selection, outliers, actions, and DRT or simulator tools |
| Status bar | Reports the current operation and result |

## Core concepts

- A **spectrum** is one cycle from one imported file, or a simulator-generated spectrum.
- A point is **active** when it is manually included and inside the frequency window. Outlier actions update the manual mask without reactivating excluded points.
- The **EEC model** is an `impedance.py` circuit string, for example `R0-L0-p(R1,CPE1)`.
- **DRT** means distribution of relaxation times. Ridge DRT and Hybrid DRT are available as analysis methods.
- **Fitted** means an EEC fit has completed for that spectrum. **DRT** means a DRT calculation is available for that spectrum.
- Project files use the `.eisfit.json` format (optionally gzip-compressed) and preserve spectra, cycle state, models, parameters, fits, and DRT results.

## Where to go next

- New to eisyFIT: read [quick-start.md](quick-start.md).
- Looking for one button or menu item: use [button-reference.md](button-reference.md).
- Automating repeated work: read [procedures-and-batch.md](procedures-and-batch.md).
- Interpreting DRT results: read [drt-and-outliers.md](drt-and-outliers.md).
- Working with fitted parameters across many spectra: read [explorers-and-export.md](explorers-and-export.md).
- Using machine-learning assistance: read [ml-processing.md](ml-processing.md).
- Changing fitting behavior: read [preferences.md](preferences.md).
