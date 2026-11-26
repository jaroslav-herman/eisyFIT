# ML processing

eisyFIT can use pretrained machine-learning pipelines to suggest EEC topology and initial parameters. ML assistance is always explicit: predictions are previewed or labelled and never silently replace your data.

## What ML can do

- Suggest an EEC model/topology for selected spectra.
- Predict initial EEC parameter values.
- Apply saved ML results to spectra.
- Fit spectra after ML suggestions are applied.

## Open the ML dialog

Use the **ML processing…** menu item in the menu bar.

The dialog lets you choose operations for the selected spectra. Typical choices are:

- infer topology/initial parameters with the available pretrained pipelines,
- apply previously saved ML results,
- fit affected spectra after applying suggestions.

## Saved pipelines per model

Each trained model (Sputtered cathode, AEM-WE, Flex) can have its own pipeline — the ordered actions plus outlier/refinement settings — saved in **Preferences → ML pipelines**.

In the ML processing dialog, enable **Apply saved pipeline for the selected model (from Preferences)**. The dialog then loads the pipeline saved for the selected model, disables the manual action list and settings, and **Run** executes exactly that saved pipeline. Switching the model while the checkbox is enabled reloads the newly selected model's pipeline.

When the checkbox is off, the dialog works manually as before: you build the action list and set the outlier/refinement values yourself.

## Load ML results

Use **File → Load ML results file…** to load a saved ML results file. The results become available to the ML workflow and can be matched to spectra by the path-independent spectrum identifier.

## Safety rules

- ML suggestions do not change spectra until you explicitly apply them.
- The GUI consumes serialized, validated predictions.
- Applying ML suggestions requires an explicit user action.
- A restoration path is retained where a suggestion replaces existing values.
- Raw measured arrays stay in the project; ML sidecars do not duplicate them.

## Practical workflow

1. Select one or more spectra in the Spectra Explorer.
2. Open **ML processing…**.
3. Choose the ML operations to run.
4. Review which spectra are available and which are unavailable.
5. Apply suggestions explicitly.
6. Fit the affected spectra.
7. Save the project.

## Related actions

- **Suggest EEC initials…** in the Fit menu or Actions group suggests initial values from compatible fitted spectra without using the ML pipelines.
- **Detect EEC parameter anomalies…** compares selected fitted parameter vectors against a calibration set.
- **Auto model: selected (Hybrid DRT)** suggests a model from Hybrid DRT analysis rather than ML.

## Correction memory: learning from your manual fixes

Some spectra are still fitted badly after the ML pipeline and need manual
corrections. The correction memory records what you do and replays it on
similar spectra — no model is trained.

1. Enable **Record my correction steps** in **Preferences → ML pipelines**.
   While on, every parameter, model, point-selection, or frequency-window
   change is appended to a sidecar next to the project,
   `<project>_user_actions.json`, keyed by a path-independent spectrum
   identifier. Raw measured arrays are never duplicated in the sidecar.
2. Correct a few badly fitted spectra by hand as usual (edit parameters or
   bounds, refit), and save the project.
3. Run **Suggest corrections…** from the menu bar (with the affected spectra
   selected, or with nothing selected to check the whole dataset). Spectra
   whose fitted parameters deviate from the smooth trend over time, voltage,
   current, cycle, and cycle mod are flagged (robust z above the configured
   **bad-fit parameter z threshold**); for each flagged spectrum, the nearest
   logged corrections in that feature space are proposed, with parameter names
   mapped structurally when the EEC topology differs.
4. Review the proposal table and confirm **Apply selected and refit** (or
   **Apply selected**). Nothing changes until you confirm; every applied
   correction is undoable with Ctrl+Z and, while recording is on, is itself
   logged with `applied_by: "replay"` plus its source corrections.

Only parameter and (structurally equivalent) circuit changes are replayed;
point masks and frequency windows are spectrum-specific and are never
force-matched onto another spectrum.

The same sidecar and replay logic are exposed to AI agents through the MCP
server — see [mcp-server.md](mcp-server.md). Agent-applied corrections are
logged with `applied_by: "mcp"` and become replay sources themselves.
