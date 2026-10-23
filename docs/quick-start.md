# Quick start

This is the shortest path from opening eisyFIT to a fitted, saved spectrum.

## 1. Start the application

From the repository directory:

```text
uv run python main.py
```

To open a BioLogic file immediately and select a cycle:

```text
uv run python main.py path/to/spectrum.mpt --cycle 1
```

On Windows, `run_gui.cmd` can also be double-clicked.

## 2. Import data

Use **File → Import data…** (`Ctrl+I`) and select one or more BioLogic `.mpt` files.

- Multiple files can be selected at once; previously loaded spectra are kept.
- If a text/CSV import dialog appears, choose the cycle column, frequency/impedance columns, and metadata columns, then press **Import**.
- The imported spectra appear in the **Spectra explorer**.

Alternatives:

- **File → Load project…** (`Ctrl+L`) restores a saved `.eisfit.json` project.
- **File → Load RelaxIS 3 project** imports a RelaxIS 3 project.

## 3. Select a spectrum and cycle

- Click a row in the **Spectra explorer** to make that spectrum active.
- Use the **Cycle** selector or the `Up`/`Down` arrow keys to move between cycles.
- `Shift+Up`/`Shift+Down` move and keep the explorer selection; `Ctrl+Up`/`Ctrl+Down` move focus without changing the selected spectra.

## 4. Choose an EEC model

In the **Fitting model** group:

1. Select a preset from the model box, or type an `impedance.py` circuit string such as `R0-L0-p(R1,CPE1)`.
2. Press **Set model**.
3. The **Circuit parameters** table updates to the parameters of that model.

To apply one model to several spectra, select them in the explorer and press **Set model for selected**.

## 5. Set the frequency window and clean points

In the **Selection** group:

- Enter **Min. freq.** and **Max. freq.** to restrict the active frequency window.
- Enable **Auto** next to Max. freq. to derive the upper limit from the spectrum.
- Press **Apply lower to selected spectra** or **Apply upper to selected spectra** to copy one limit to selected spectra.

To remove bad points manually:

1. In the plot controls, press **Edit points: Off** until it reads **Edit points: On**.
2. Click points in the Nyquist/Bode plot to toggle them between included and excluded.
3. Optionally enable **Edit points and fit: Off** to fit automatically after each point change.
4. Press **Active points** to reset the view to active points, or **Reset points** to include all points and clear detected outliers.

## 6. Initialize parameters

Use one of these before fitting:

- **Initial values** — run bayes-drt2 ridge analysis on active points, apply its outlier mask, and initialize the EEC fit from the ohmic resistance, inductance, and strongest DRT peaks.
- **Suggest EEC initials…** — calculate initial values from compatible previously fitted spectra and apply them, optionally fitting immediately.
- Edit the **Circuit parameters** table directly when you already know good starting values.

Outlier detection never runs automatically.

## 7. Fit

- Press **Fit spectrum** (`Alt+S`) to fit the displayed spectrum with the parameter-table values.
- Select spectra in the explorer and press **Fit selected** to fit each selected spectrum using its own model and initial parameters.
- The fitted curve appears in the plot when **Show EEC fit** is enabled.
- The explorer **Fitted** column marks spectra with a completed fit.

## 8. Review and save

- Compare measured and fitted points at identical frequencies; residual connectors show the difference.
- Press **Stop** to cancel a running fit or batch operation.
- Save the whole project with **File → Save project…** (`Ctrl+S`).
- Export results from the **Export** menu, or read [explorers-and-export.md](explorers-and-export.md).

## Typical first workflow

```text
Import data → select spectrum → set model → set frequency window
→ Initial values → Fit spectrum → review fit → Save project
```
