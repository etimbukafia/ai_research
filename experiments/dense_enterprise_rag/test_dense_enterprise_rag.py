"""Tests for the controlled dense enterprise RAG experiment."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from build_graph import build_graph, generate_cases, generate_documents, validate_graph
from retrieval import CONDITIONS, graph_from_export


DIRECTORY = Path(__file__).resolve().parent


def read_json(name: str):
    return json.loads((DIRECTORY / name).read_text(encoding="utf-8"))


class DenseEnterpriseRAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.documents = read_json("documents.json")["documents"]
        cls.cases = read_json("cases.json")["cases"]
        cls.graph_export = read_json("graph.json")
        cls.results = read_json("results.json")
        cls.live = read_json("live_results.json")
        cls.extraction = read_json("extraction_results.json")

    def test_document_contract(self) -> None:
        self.assertEqual(len(self.documents), 96)
        self.assertEqual(len({doc["document_id"] for doc in self.documents}), 96)
        by_id = {doc["document_id"]: doc for doc in self.documents}
        self.assertEqual(by_id["spec-17"]["product_ids"], ["atlas_x200"])
        self.assertEqual(by_id["spec-17"]["firmware_versions"], ["4.2"])
        self.assertEqual(by_id["spec-17"]["regions"], ["europe"])
        self.assertEqual(by_id["spec-17"]["status"], "Approved")
        self.assertEqual(by_id["spec-17"]["authority"], "product_safety")
        pressure = next(f for f in by_id["spec-17"]["facts"] if f["name"] == "pressure_limit")
        self.assertEqual(pressure, {"name": "pressure_limit", "value": 220, "unit": "kPa"})
        self.assertEqual(by_id["spec-12"]["status"], "Superseded")
        self.assertEqual(by_id["spec-19"]["status"], "Draft")
        self.assertEqual(by_id["bulletin-08"]["document_role"], "Informational")
        self.assertEqual(by_id["spec-27"]["product_ids"], ["atlas_x210"])

    def test_case_contract(self) -> None:
        self.assertEqual(len(self.cases), 32)
        self.assertEqual(len({case["pair_id"] for case in self.cases}), 16)
        self.assertEqual(sum(case["is_holdout"] for case in self.cases), 8)
        for case_class in {
            "version_and_supersession",
            "region_and_entity_scope",
            "authority_and_status",
            "missing_or_ambiguous_evidence",
        }:
            self.assertEqual(
                sum(case["case_class"] == case_class for case in self.cases), 8
            )
        dense = [case for case in self.cases if case["is_dense_overlap"]]
        self.assertEqual(len(dense), 16)
        running = next(case for case in self.cases if case["case_id"] == "case-01-dense")
        self.assertEqual(
            running["question"],
            "What pressure limit applies to the Atlas X200 in Europe with firmware 4.2?",
        )
        self.assertEqual(running["applicable_document_id"], "spec-17")
        self.assertEqual(running["required_fact"]["value"], 220)

    def test_graph_export_and_schema(self) -> None:
        graph = graph_from_export(self.graph_export)
        validate_graph(graph)
        self.assertEqual(self.graph_export["node_count"], graph.number_of_nodes())
        self.assertEqual(self.graph_export["edge_count"], graph.number_of_edges())
        self.assertGreater(graph.number_of_nodes(), 96)
        self.assertGreater(graph.number_of_edges(), 96)
        self.assertIn("document-spec-17", graph.nodes)
        self.assertIn("fact-spec-17-pressure-limit", graph.nodes)
        self.assertTrue(
            any(
                attrs.get("type") == "SUPERSEDES"
                for attrs in graph.get_edge_data("document-spec-17", "document-spec-12", default={}).values()
            )
        )

    def test_generators_are_stable(self) -> None:
        documents = generate_documents()
        cases = generate_cases(documents)
        self.assertEqual(documents, self.documents)
        self.assertEqual(cases, self.cases)
        self.assertEqual(
            read_json("graph_schema.json")["validation_rules"],
            [
                "Reject unknown node or edge types.",
                "Reject edges without source_id.",
                "Reject time-varying edges without valid_from and valid_to.",
                "Reject numeric facts without a unit.",
                "Reject specification facts without an authority.",
                "Reject SUPERSEDES cycles.",
            ],
        )

    def test_graph_rejects_invalid_records(self) -> None:
        graph = graph_from_export(self.graph_export)
        invalid_node = copy.deepcopy(graph)
        invalid_node.add_node("bad-node", type="Unknown")
        with self.assertRaises(ValueError):
            validate_graph(invalid_node)

        invalid_edge = copy.deepcopy(graph)
        invalid_edge.add_edge(
            "product-atlas-x200",
            "document-spec-17",
            type="Unknown",
            source_id="spec-17",
        )
        with self.assertRaises(ValueError):
            validate_graph(invalid_edge)

        missing_source = copy.deepcopy(graph)
        missing_source.add_edge(
            "product-atlas-x200",
            "document-spec-17",
            type="APPLIES_TO",
            valid_from="2026-01-01",
            valid_to="2027-01-01",
        )
        with self.assertRaises(ValueError):
            validate_graph(missing_source)

        cycle = copy.deepcopy(graph)
        cycle.add_edge(
            "document-spec-12",
            "document-spec-17",
            type="SUPERSEDES",
            source_id="spec-12",
            valid_from="2022-01-01",
            valid_to="2025-01-01",
        )
        with self.assertRaises(ValueError):
            validate_graph(cycle)

    def test_all_conditions_have_the_same_cases(self) -> None:
        self.assertEqual(set(self.results["conditions"]), set(CONDITIONS))
        for condition in CONDITIONS:
            self.assertEqual(self.results["conditions"][condition]["row_count"], 32)
            metrics = self.results["conditions"][condition]["metrics"]
            self.assertEqual(metrics["overall"]["case_count"], 32)
            self.assertEqual(
                metrics["overall"]["strict_applicability_accuracy"]["total"], 24
            )
            self.assertEqual(metrics["overall"]["correct_abstention_rate"]["total"], 8)
            self.assertGreaterEqual(metrics["overall"]["compression"]["count"], 24)
        self.assertEqual(
            self.results["conditions"]["graph_guided_hybrid"]["metrics"]["overall"]["compression"]["count"],
            24,
        )

    def test_graph_condition_preserves_the_running_path(self) -> None:
        traces = [
            json.loads(line)
            for line in (DIRECTORY / "traces.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(traces), 128)
        running = next(
            trace
            for trace in traces
            if trace["case_id"] == "case-01-dense"
            and trace["condition"] == "graph_guided_hybrid"
        )
        self.assertEqual(running["selected_document_id"], "spec-17")
        self.assertEqual(
            running["answer"]["value"],
            220,
        )
        self.assertEqual(running["answer"]["unit"], "kPa")
        self.assertEqual(
            running["selected_evidence_path"],
            [
                "product-atlas-x200",
                "firmware-4-2",
                "region-europe",
                "document-spec-17",
                "fact-spec-17-pressure-limit",
                "authority-product-safety",
            ],
        )

    def test_optional_artifacts_are_truthful(self) -> None:
        self.assertIn(self.live["status"], {"not_run", "partial", "completed"})
        self.assertEqual(self.live["planned_requests"], 64)
        self.assertEqual(self.live["rpm_limit"], 15)
        self.assertEqual(self.live["interval_seconds"], 5.0)
        self.assertEqual(self.extraction["document_count"], 16)
        if self.live["status"] == "not_run":
            self.assertEqual(self.live["request_count"], 0)
            self.assertIsNotNone(self.live["reason"])
        if self.extraction["status"] == "not_run":
            self.assertEqual(self.extraction["request_count"], 0)
            self.assertIsNotNone(self.extraction["reason"])

    def test_no_api_key_in_artifacts(self) -> None:
        for path in DIRECTORY.iterdir():
            if path.is_file() and path.suffix in {".py", ".json", ".md", ".txt", ".jsonl"}:
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("AI" + "za" + "Sy", text, path.name)

    def test_manifest_records_source_and_policy(self) -> None:
        manifest = read_json("run_manifest.json")
        self.assertEqual(manifest["document_count"], 96)
        self.assertEqual(manifest["case_count"], 32)
        self.assertFalse(manifest["docker_required"])
        self.assertFalse(manifest["external_graph_database_required"])
        self.assertIn("GOOGLE_API_KEY", manifest["live_key_policy"])
        for filename in manifest["source_sha256"]:
            self.assertTrue((DIRECTORY / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main(verbosity=2)
