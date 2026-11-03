"""Privacy checks for prospect lookup, profile caching, and scoring."""

import importlib
import json
import os
import unittest
from unittest.mock import patch

from langsmith import trace, tracing_context

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")

data_service = agent.data_service


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(tracing_context(enabled=False))
        self.enterContext(patch("langsmith.run_trees.RunTree.post"))
        self.enterContext(patch("langsmith.run_trees.RunTree.patch"))
        self.cache = patch.dict(data_service._PROFILES, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.prospect_id = next(iter(data_service.PROSPECTS))
        self.offering_id = next(iter(data_service.OFFERINGS))
        self.record = data_service.PROSPECTS[self.prospect_id]
        self.arguments = {"prospect_id": self.prospect_id, "offering_id": self.offering_id}
        self.result = agent.ProspectScore(
            score=75, justification="Fit", rubric_breakdown={
                "revenue_fit": 80, "tech_stack_match": 70, "segment_fit": 75,
            },
        )

    def assert_private(self, value):
        serialized = json.dumps(value, default=str)
        self.assertNotIn("billing_qualification", serialized)
        for secret in self.record["billing_qualification"].values():
            self.assertNotIn(secret, serialized)

    def test_contact_allow_list_and_missing_records(self):
        with patch.dict(self.record, {"future_private_field": "private"}):
            contact = agent.get_prospect.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(contact, {"found": True, "prospect": {
            "prospect_id": self.prospect_id,
            **{key: self.record[key] for key in (
                "name", "email", "disqualified", "annual_revenue", "enrichment_source"
            )},
        }})
        self.assert_private(contact)
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}),
                         {"prospect": None, "found": False})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
                         {"prospect_profile": None, "found": False})
        self.assertEqual(data_service.get_profile_from_db("missing"), {"prospect_profile": None})

    def test_lookup_and_scoring_trace_boundaries(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                data_service._PROFILES.clear()
                if cached:
                    data_service._PROFILES[self.prospect_id] = {
                        **self.record, "prospect_id": self.prospect_id, "tech_stack": ["stale"],
                    }
                with patch.object(agent, "_scoring_llm") as llm:
                    llm.invoke.return_value = self.result
                    with tracing_context(enabled="local"):
                        with trace("lookup and scoring") as root:
                            contact = agent.get_prospect.invoke({"prospect_id": self.prospect_id})
                            profile = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
                            score = agent.score_prospect.invoke(self.arguments)
                    self.assertEqual(score, self.result.model_dump())
                    self.assert_private(contact)
                    self.assert_private(profile)
                    self.assertEqual(profile["prospect_profile"], {
                        "prospect_id": self.prospect_id,
                        **{key: value for key, value in self.record.items()
                           if key != "billing_qualification"},
                        "tech_stack": ["stale"] if cached else self.record["tech_stack"],
                    })
                    self.assert_private(llm.invoke.call_args.args[0])
                    prompt = llm.invoke.call_args.args[0][1]["content"]
                    scored_profile = json.loads(prompt.split("Prospect profile:\n", 1)[1])
                    self.assertEqual(scored_profile["tech_stack"], self.record["tech_stack"])
                    self.assertEqual(scored_profile["annual_revenue"], self.record["annual_revenue"])
                    self.assertIn(json.dumps(data_service.get_offering(self.offering_id), indent=2), prompt)
                    runs = []
                    pending = list(root.child_runs)
                    while pending:
                        run = pending.pop()
                        runs.append(run)
                        pending.extend(run.child_runs)
                        self.assert_private(run.inputs)
                        self.assert_private(run.outputs)
                    names = {run.name for run in runs}
                    self.assertIn("get_profile_from_db", names)
                    if not cached:
                        self.assertIn("save_profile_to_db", names)
                        self.assert_private(data_service._PROFILES[self.prospect_id])
                    scoring = next(run for run in runs if run.name == "score_prospect")
                    self.assertEqual(scoring.inputs, self.arguments)
                    self.assertIn("billing_qualification", self.record)

    def test_scoring_schema_and_invalid_records(self):
        self.assertEqual(set(agent.score_prospect.args), {"prospect_id", "offering_id"})
        with patch.object(agent, "_scoring_llm") as llm:
            for offering in (None, {}, {"required_tech_stack": ["AWS"], "description": "Fit"},
                             {"min_annual_revenue": 1, "description": "Fit"},
                             {"required_tech_stack": ["AWS"], "min_annual_revenue": 1}):
                with self.subTest(offering=offering):
                    with patch.object(data_service, "get_offering", return_value=offering):
                        self.assertEqual(agent.score_prospect.invoke(self.arguments), {
                            "score": None, "error": "Cannot score without a valid offering.",
                        })
            self.assertEqual(agent.score_prospect.invoke({**self.arguments, "prospect_id": "missing"}), {
                "score": None, "error": "Cannot score without a valid prospect.",
            })
            llm.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
