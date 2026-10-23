# Procedures and batch fitting

Batch fitting and saved procedures let you repeat the same sequence of actions across many spectra.

## Batch fit from the Fit menu

The **Fit** menu can fit spectra in explorer order:

- **Batch down** fits spectra below the current spectrum, using each result to initialize the next.
- **Batch up** fits spectra above the current spectrum, using each result to initialize the next.
- **Batch fit selected down**, **Batch fit selected up**, and **Batch selected up and down** operate on the spectra selected in the Spectra Explorer.
- **Batch down to metadata value…** and **Batch up to metadata value…** stop when a numeric explorer metadata column reaches a chosen target value. The batch uses the last clicked numeric explorer column.

Batch operations:

- run in the background,
- keep the GUI responsive,
- update each spectrum's fit and initial parameters,
- can be cancelled with **Stop**.

## Recommended batch workflow

1. Import all related `.mpt` files.
2. Select the spectra in the Spectra Explorer in the order you want to fit.
3. Set a suitable EEC model with **Set model for selected**.
4. Set a common frequency window and apply it to selected spectra.
5. Fit one representative spectrum first.
6. Use **Batch fit selected down**, **Batch fit selected up**, or **Batch selected up and down**.
7. Review the explorer **Fitted** markers and the fitted curves.
8. Save the project.

## Procedure builder

Open **Procedures → Procedure builder…** to save reusable blocks and chain them into named procedures.

### Blocks

A block is a named sequence of steps. Build a block by choosing an action and, when needed, its argument:

| Action | Argument | Effect |
| --- | --- | --- |
| Set frequency limits for selected | `minimum,maximum` | Applies a frequency window to selected spectra. |
| Fit current spectrum | none | Fits the displayed spectrum. |
| Fit selected | none | Fits selected spectra using each spectrum's model and initial parameters. |
| Batch fit selected | `up`, `down`, or `up and down` | Batch-fits the selected spectra. |
| Initialize from ridge | none | Runs ridge analysis and initializes EEC parameters for selected spectra. |
| Set model for selected | circuit string | Applies an EEC model to selected spectra. |

Use **Save block** to store the current step list under a block name.

### Procedures

A procedure is an ordered list of blocks with optional per-step arguments. Save it under a procedure name, then run it on the current selection.

- **Save procedure** stores the procedure in the builder.
- **Save to file** / **Load from file** share procedures as JSON.
- **Save to Preferences** / **Load from Preferences** keep procedures across application restarts.
- **Run** executes the procedure steps in order and reports failures without losing the rest of the project.

## Tips

- Fit one spectrum manually before starting a batch so the model and initial values are known to be reasonable.
- Keep the frequency window consistent across related spectra before batch fitting.
- Use metadata-limited batches when a series should stop at a specific voltage, current, time, or cycle value.
- Save the project after a successful batch; fits are stored in the project file.
