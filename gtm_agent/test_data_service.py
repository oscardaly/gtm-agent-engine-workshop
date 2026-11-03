"""Tests for prospect technology updates and downstream reads."""

import json
import os
import unittest
from copy import deepcopy
from unittest.mock import patch

from . import data_service

with patch.dict(os.environ, OPENAI_API_KEY="test", LANGSMITH_TRACING="false"):
    from . import gtm_agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, LANGSMITH_TRACING="false"))
        self.enterContext(patch.dict(data_service.PROSPECTS, deepcopy(data_service.PROSPECTS)))
        self.enterContext(patch.dict(data_service._PROFILES, {}, clear=True))
        self.prospect_id = "LEAD-39002"
        self.technology = "Terraform"
        self.assertNotIn(self.technology, data_service.fetch_tech_stack(self.prospect_id))

    def test_update_persists_technology(self):
        result = data_service.update_prospect_info(self.prospect_id, self.technology)

        self.assertTrue(result["updated"])
        self.assertTrue(result["found"])
        self.assertIn(self.technology, result["tech_stack"])
        self.assertEqual(result["tech_stack"], data_service.fetch_tech_stack(self.prospect_id))

    def test_update_refreshes_cached_profile(self):
        before = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertNotIn(self.technology, before["prospect_profile"]["tech_stack"])

        data_service.update_prospect_info(self.prospect_id, self.technology)
        after = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        cached = data_service.get_profile_from_db(self.prospect_id)["prospect_profile"]

        self.assertIn(self.technology, after["prospect_profile"]["tech_stack"])
        self.assertEqual(after["prospect_profile"], cached)

    def test_unknown_prospect_leaves_records_and_cache_unchanged(self):
        records = deepcopy(data_service.PROSPECTS)
        profiles = deepcopy(data_service._PROFILES)

        result = data_service.update_prospect_info("unknown-prospect", self.technology)

        self.assertEqual(result, {"updated": False, "found": False})
        self.assertEqual(data_service.PROSPECTS, records)
        self.assertEqual(data_service._PROFILES, profiles)

    def test_update_then_score_uses_persisted_stack_with_stale_profile(self):
        profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        offering = data_service.get_offering("OFFER-10004")
        gtm_agent.update_prospect_info.invoke({
            "prospect_id": self.prospect_id, "technology": self.technology,
        })

        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            gtm_agent.score_prospect.invoke({
                "prospect_profile": profile["prospect_profile"], "offering": offering,
            })

        messages = scoring_llm.invoke.call_args.args[0]
        scored_profile = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
        self.assertIn(self.technology, scored_profile["tech_stack"])
        self.assertFalse(set(offering["required_tech_stack"]) - set(scored_profile["tech_stack"]))
