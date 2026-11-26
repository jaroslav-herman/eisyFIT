"""Tests for the eisyFIT MCP server (mcp_server.py).

The tool functions are tested in-process; a separate stdio smoke test spawns
``python -m mcp_server`` and drives it through the MCP client SDK.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import mcp_server
from eis_project import save_project_file
from eis_services import load_cycle, load_project_from_dataframe
from ml.user_actions import (
    append_user_actions,
    load_user_actions,
    make_action_entry,
    user_actions_path_for,
)

try:
    import mcp  # noqa: F401

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


def _make_project(root: Path, cycles: int = 6) -> Path:
    """Create a small saved project: circuit R0, one planted bad fit (cycle 4)."""
    freqs = np.geomspace(1e4, 1e-1, 8)
    rows = []
    for cycle in range(1, cycles + 1):
        r0 = 2.0 + 0.1 * cycle
        for frequency in freqs:
            rows.append(
                {
                    "freq_hz": float(frequency),
                    "cycle_number": cycle,
                    "re_zwe_ce_ohm": r0,
                    "minus_im_zwe_ce_ohm": -0.01,
                    "ewe_ece_v": 1.0 + 0.01 * cycle,
                    "i_ma": 2.0,
                    "time_s": 10.0 * cycle,
                }
            )
    dataframe = pd.DataFrame(rows)
    loaded = load_project_from_dataframe(dataframe, root / "source.mpt", 1, "cell", "R0")
    state = loaded.state
    for cycle in state.available_cycles:
        cycle_state = load_cycle(dataframe, cycle, "cell")
        cycle_state.circuit = "R0"
        cycle_state.parameters = state.parameters_for(cycle)
        fitted = 2.0 + 0.1 * cycle + (5.0 if cycle == 4 else 0.0)
        cycle_state.fit_parameters = np.array([fitted])
        cycle_state.fit_at_data_impedance = np.full(
            cycle_state.frequency_hz.size, complex(fitted, -0.01)
        )
        state.cycles[cycle] = cycle_state
    project_path = root / "sample.eisfit.json"
    save_project_file(state, project_path, datasets=[(loaded.dataset_id, state, dataframe)])
    return project_path


def _plant_user_action(project_path: Path, dataset_id: str) -> None:
    entry = make_action_entry(
        applied_by="user",
        action="cycle 3: parameters",
        dataset_id=dataset_id,
        dataset_label="sample",
        control="cell",
        spectrum_id="eisfit-spectrum-v1:planted",
        cycle=3,
        circuit="R0",
        metadata={
            "cycle": 3,
            "time_s": 30.0,
            "potential_v": 1.03,
            "i_ma": 2.0,
            "Cycle mod 15": 3,
        },
        change={
            "cycle": "3",
            "parameters": [{"name": "R0", "field": "initial", "old": 2.3, "new": 9.9}],
            "circuit": None,
            "frequency_window": None,
            "points_excluded": [],
            "points_included": [],
            "fit_changed": False,
        },
    )
    append_user_actions(user_actions_path_for(project_path), [entry], source_project=str(project_path))


class McpToolTests(unittest.TestCase):
    def setUp(self) -> None:
        mcp_server._SESSION.projects.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.project_path = _make_project(self.root)
        self.project = str(self.project_path)

    def tearDown(self) -> None:
        mcp_server._SESSION.projects.clear()
        self._tmp.cleanup()

    def test_open_and_list_spectra(self):
        opened = mcp_server.open_project(self.project)
        self.assertEqual(opened["file"], "sample.eisfit.json")
        self.assertEqual(len(opened["datasets"]), 1)
        listed = mcp_server.list_spectra(self.project)
        self.assertEqual(listed["count"], 6)
        first = listed["spectra"][0]
        self.assertTrue(first["identifier"].startswith("eisfit-spectrum-v1:"))
        self.assertTrue(first["has_fit"])
        self.assertEqual(first["circuit"], "R0")
        self.assertIn(self.project, mcp_server.list_projects()["projects"][0]["project"])

    def test_resolve_by_identity_and_spectrum_id(self):
        listed = mcp_server.list_spectra(self.project)
        by_identity = mcp_server.get_parameters(self.project, f"{listed['spectra'][0]['dataset_id']}::cycle-2")
        self.assertEqual(by_identity["identity"].endswith("cycle-2"), True)
        by_spectrum_id = mcp_server.get_parameters(self.project, listed["spectra"][1]["identifier"])
        self.assertEqual(by_spectrum_id["identity"].endswith("cycle-2"), True)
        self.assertEqual(by_spectrum_id["parameters"][0]["name"], "R0")

    def test_get_fit_reports_planted_fit(self):
        listed = mcp_server.list_spectra(self.project)
        target = next(s for s in listed["spectra"] if s["cycle"] == 4)
        fit = mcp_server.get_fit(self.project, target["identifier"])
        self.assertTrue(fit["has_fit"])
        self.assertAlmostEqual(fit["fitted_parameters"][0]["value"], 7.4)

    def test_set_parameters_dry_run_then_confirm_logs_sidecar(self):
        identity = mcp_server.list_spectra(self.project)["spectra"][0]["identity"]
        dry = mcp_server.set_parameters(
            self.project, identity, [{"name": "R0", "initial": 5.5}]
        )
        self.assertTrue(dry["dry_run"])
        self.assertTrue(dry["confirm_required"])
        unchanged = mcp_server.get_parameters(self.project, identity)
        self.assertNotEqual(unchanged["parameters"][0]["initial"], 5.5)
        applied = mcp_server.set_parameters(
            self.project, identity, [{"name": "R0", "initial": 5.5}], confirm=True
        )
        self.assertTrue(applied["applied"])
        changed = mcp_server.get_parameters(self.project, identity)
        self.assertEqual(changed["parameters"][0]["initial"], 5.5)
        entries = load_user_actions(user_actions_path_for(self.project_path))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["applied_by"], "mcp")
        self.assertEqual(entries[0]["changes"]["parameters"][0]["new"], 5.5)

    def test_set_parameters_rejects_invalid_bounds(self):
        identity = mcp_server.list_spectra(self.project)["spectra"][0]["identity"]
        result = mcp_server.set_parameters(
            self.project, identity, [{"name": "R0", "initial": 1e9}], confirm=True
        )
        self.assertFalse(result["ok"])
        result = mcp_server.set_parameters(
            self.project, identity, [{"name": "NOPE", "initial": 1.0}]
        )
        self.assertFalse(result["ok"])

    def test_point_mask_and_window_and_outliers(self):
        spectra = mcp_server.list_spectra(self.project)["spectra"]
        identity = spectra[0]["identity"]
        dry = mcp_server.set_point_mask(self.project, identity, exclude=[0, 1])
        self.assertTrue(dry["dry_run"])
        self.assertEqual(dry["changes"]["points_excluded"], [0, 1])
        applied = mcp_server.set_point_mask(self.project, identity, exclude=[0, 1], confirm=True)
        self.assertTrue(applied["applied"])
        detail = mcp_server.get_spectrum(self.project, identity, include_arrays=True)
        self.assertEqual(detail["active_points"], detail["point_count"] - 2)
        window = mcp_server.set_frequency_window(self.project, identity, 1.0, 100.0, confirm=True)
        self.assertTrue(window["applied"])
        outliers = mcp_server.apply_outliers(self.project, identity, [2], confirm=True)
        self.assertTrue(outliers["applied"])
        self.assertEqual(outliers["excluded_indices"], [2])

    def test_flag_propose_apply_corrections(self):
        listed = mcp_server.list_spectra(self.project)
        dataset_id = listed["spectra"][0]["dataset_id"]
        _plant_user_action(self.project_path, dataset_id)
        flags = mcp_server.flag_bad_fits(self.project, z_threshold=4.0)
        flagged = {f["identity"] for f in flags["flagged"]}
        self.assertIn(f"{dataset_id}::cycle-4", flagged)
        proposals = mcp_server.propose_corrections(self.project, z_threshold=4.0)
        self.assertEqual(proposals["logged_corrections"], 1)
        self.assertTrue(proposals["proposals"])
        proposal = next(
            p for p in proposals["proposals"] if p["identity"] == f"{dataset_id}::cycle-4"
        )
        self.assertEqual(proposal["changes"][0]["new"], 9.9)
        dry = mcp_server.apply_corrections(self.project, confirm=False)
        self.assertTrue(dry["dry_run"])
        applied = mcp_server.apply_corrections(
            self.project,
            proposal_identities=[f"{dataset_id}::cycle-4"],
            confirm=True,
        )
        self.assertEqual(len(applied["applied"]), 1)
        params = mcp_server.get_parameters(self.project, f"{dataset_id}::cycle-4")
        self.assertEqual(params["parameters"][0]["initial"], 9.9)
        entries = load_user_actions(user_actions_path_for(self.project_path))
        self.assertEqual(len(entries), 2)
        mcp_entry = entries[-1]
        self.assertEqual(mcp_entry["applied_by"], "mcp")
        self.assertTrue(mcp_entry.get("replay_sources"))

    def test_fit_spectrum_and_save_with_backup(self):
        identity = mcp_server.list_spectra(self.project)["spectra"][0]["identity"]
        result = mcp_server.fit_spectrum(self.project, identity, timeout_seconds=30.0)
        self.assertIn("rmse", result)
        fit = mcp_server.get_fit(self.project, identity)
        self.assertTrue(fit["has_fit"])
        saved = mcp_server.save_project(self.project)
        self.assertTrue(Path(saved["saved"]).is_file())
        # the project file already existed, so the previous version is backed up
        self.assertTrue(Path(saved["backup"]).is_file())
        mcp_server.set_parameters(
            self.project, identity, [{"name": "R0", "initial": 3.3}], confirm=True
        )
        saved_again = mcp_server.save_project(self.project)
        self.assertTrue(Path(saved_again["backup"]).is_file())
        reloaded = mcp_server.reload_project(self.project)
        params = mcp_server.get_parameters(self.project, identity)
        self.assertEqual(params["parameters"][0]["initial"], 3.3)
        self.assertIn("datasets", reloaded)

    def test_close_project(self):
        mcp_server.open_project(self.project)
        closed = mcp_server.close_project(self.project)
        self.assertEqual(closed["closed"], self.project)
        self.assertEqual(mcp_server.list_projects()["projects"], [])


@unittest.skipUnless(MCP_AVAILABLE, "mcp SDK not installed")
class StdioSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        mcp_server._SESSION.projects.clear()

    def test_stdio_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_path = _make_project(Path(tmp))
            asyncio.run(self._roundtrip(str(project_path)))

    async def _roundtrip(self, project: str) -> None:
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server"],
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {tool.name for tool in tools.tools}
                for expected in (
                    "open_project",
                    "list_spectra",
                    "fit_spectrum",
                    "set_parameters",
                    "flag_bad_fits",
                    "apply_corrections",
                    "save_project",
                ):
                    self.assertIn(expected, names)

                opened = await session.call_tool("open_project", {"project": project})
                payload = self._structured(opened)
                self.assertEqual(payload["file"], "sample.eisfit.json")

                listed = await session.call_tool("list_spectra", {"project": project})
                spectra = self._structured(listed)["spectra"]
                self.assertEqual(len(spectra), 6)

                params_result = await session.call_tool(
                    "get_parameters", {"project": project, "identifier": spectra[0]["identity"]}
                )
                self.assertEqual(self._structured(params_result)["parameters"][0]["name"], "R0")

                dry = await session.call_tool(
                    "set_parameters",
                    {
                        "project": project,
                        "identifier": spectra[0]["identity"],
                        "changes": [{"name": "R0", "initial": 4.4}],
                    },
                )
                self.assertTrue(self._structured(dry)["dry_run"])

                resources = await session.list_resources()
                uris = {str(resource.uri) for resource in resources.resources}
                self.assertIn("eisyfit://projects", uris)
                contents = await session.read_resource(
                    "eisyfit://projects/sample.eisfit.json/user-actions"
                )
                sidecar = json.loads(contents.contents[0].text)
                self.assertEqual(sidecar["format"], "eis-fitting-user-actions")

                prompts = await session.list_prompts()
                prompt_names = {prompt.name for prompt in prompts.prompts}
                self.assertIn("correct_bad_fits", prompt_names)
                prompt = await session.get_prompt(
                    "correct_bad_fits", {"project": project}
                )
                self.assertTrue(prompt.messages)

    @staticmethod
    def _structured(result) -> dict:
        assert not getattr(result, "isError", False), result.content
        structured = getattr(result, "structuredContent", None)
        if structured:
            return structured
        return json.loads(result.content[0].text)


if __name__ == "__main__":
    unittest.main()
