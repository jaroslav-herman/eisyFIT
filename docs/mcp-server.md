# MCP server: drive eisyFIT from any AI agent

`mcp_server.py` exposes eisyFIT as an [MCP](https://modelcontextprotocol.io) server over
**stdio**: any MCP-capable client (Mistral Vibe, Claude Desktop, VS Code Copilot, …) spawns
`eisyfit-mcp` as a local subprocess and can open projects, inspect spectra, run fits and DRT,
and replay recorded corrections — without opening the GUI. The server is model-agnostic: it
only speaks the MCP protocol, so the same binary works with every MCP client.

The server never imports the GUI module; it drives the same service layer (`eis_services.py`,
`eis_project.py`, `ml/`) as the editor, so all repository invariants hold: mutations that
invalidate a fit clear it, spectra are matched by `spectrum_identifier()`, saves are atomic,
and point masks / frequency windows are never force-matched across spectra.

## Installation

```powershell
# inside the eisyFIT environment, adds the official MCP Python SDK
.\.venv\Scripts\python.exe -m pip install "mcp>=1.10"
# or: uv pip install --optional mcp .
```

Run it directly for testing:

```powershell
.\.venv\Scripts\python.exe -m mcp_server            # stdio
.\.venv\Scripts\eisyfit-mcp.exe --project path	o\project.eisfit.json.gz
```

## Client configuration

### Mistral Vibe (user-global, applies to every project)

Add to `~/.vibe/config.toml` (or run `vibe mcp add eisyfit --transport stdio --command <python> --arg -m --arg mcp_server`):

```toml
[[mcp_servers]]
name = "eisyfit"
transport = "stdio"
command = "C:\\path\\to\\eisyFIT\\.venv\\Scripts\\python.exe"
args = ["-m", "mcp_server"]
```

In Vibe, set the mutating tools (`set_parameters`, `set_point_mask`, `set_frequency_window`,
`apply_outliers`, `apply_corrections`, `save_project`) to require approval in the session —
this is the second confirmation layer on top of the tools' own `confirm=true` gate.

### Other clients

Any MCP client that can spawn a stdio command works unchanged; use the same `command`/`args`
(e.g. `.vscode/mcp.json` for VS Code: `{"mcpServers": {"eisyfit": {"command": "...", "args": ["-m", "mcp_server"]}}}`).

## Tools

Every tool takes `project` — the path you passed to `open_project` — and spectra are addressed
by `identifier`: a `spectrum_identifier` hash from `list_spectra`, a `dataset_id::cycle-N`
identity, or a bare cycle number.

| Tool | Purpose |
|---|---|
| `open_project` / `list_projects` / `close_project` / `reload_project` | session lifecycle (also opens raw `.mpt`/`.txt` data files) |
| `list_spectra` / `get_spectrum` / `get_parameters` / `get_fit` | inspection (arrays optional and subsampled) |
| `fit_spectrum` / `batch_fit` / `refine_fit` | EEC fitting (per-fit timeout, optional refine) |
| `analyze_outliers` / `apply_outliers` | outlier search (dry-run) and application (gated) |
| `hybrid_drt` / `lin_kk_residuals` / `suggest_eec_parameters` | DRT, KK residuals, neighbor-based initials |
| `set_parameters` / `set_point_mask` / `set_frequency_window` | gated edits: dry-run diff unless `confirm=true` |
| `flag_bad_fits` / `propose_corrections` / `apply_corrections` | correction memory (see below) |
| `list_user_actions` / `save_project` | read the sidecar; atomic save with a `.bak` of the previous file |

Resources: `eisyfit://projects` (open projects) and
`eisyfit://projects/<project-file-name>/user-actions` (the recorded-corrections sidecar).
Prompt: `correct_bad_fits` (the flag → review → apply → refit → save workflow).

## Safety model

- Mutating tools are **dry-run by default** and return the exact diff; they only change state
  with `confirm=true`. In Vibe/Claude Desktop the client UI approval is a second layer.
- Applied changes are appended to the project's `<project>_user_actions.json` sidecar with
  `applied_by: "mcp"`, so GUI-recorded (`"user"`), replayed (`"replay"`), and agent-applied
  steps share one audit trail — and the agent's own fixes become replay sources later.
- `save_project` writes atomically and keeps a `<file>.bak` of the previous version.

## Correction memory over MCP

This generalizes the in-GUI "Suggest corrections…" flow: enable **Record my correction
steps** in Preferences → ML pipelines, fix a few bad spectra by hand, save; then from any agent:

```
flag_bad_fits(project, z_threshold=4.0)
propose_corrections(project, z_threshold=4.0)
apply_corrections(project, proposal_identities=[...], confirm=true, refit=true)
save_project(project)
```

Proposals replay only parameter/circuit changes from structurally compatible logged entries —
never point masks or frequency windows.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_mcp_server -v
```

covers the tools in-process plus a stdio round-trip that spawns `python -m mcp_server` and
calls tools/resources/prompts through the MCP client SDK.
