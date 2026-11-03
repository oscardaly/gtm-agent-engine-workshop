import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage

directory = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "gtm_agent_under_test", directory / "gtm_agent.py",
    submodule_search_locations=[str(directory)],
)
agent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = agent
with patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent") as create_agent:
    spec.loader.exec_module(agent)


class RunAgentTests(unittest.TestCase):
    def setUp(self):
        self.rep = agent.REP_IDS[0]

    def test_known_rep_context_precedes_email_request(self):
        request = "Email LEAD-12853 a hello."
        with patch.object(agent.gtm_agent, "invoke", return_value={
            "messages": [AIMessage(content="Sent")],
        }) as invoke:
            result = agent.run_agent(request, user_id=self.rep["rep_id"])
        messages = invoke.call_args.args[0]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn(self.rep["name"], messages[0]["content"])
        self.assertIn(self.rep["email"], messages[0]["content"])
        self.assertEqual(messages[1:], [{"role": "user", "content": request}])
        self.assertEqual(result["reply"], "Sent")
        self.assertEqual(invoke.call_args.kwargs["config"]["metadata"]["user_id"], self.rep["rep_id"])

    def test_unknown_rep_proceeds_without_identity(self):
        request = "Look up OFFER-10001."
        with patch.object(agent.gtm_agent, "invoke", return_value={
            "messages": [AIMessage(content="Found")],
        }) as invoke:
            result = agent.run_agent(request, user_id="unknown-rep")
        self.assertEqual(invoke.call_args.args[0]["messages"], [
            {"role": "user", "content": request},
        ])
        self.assertEqual(result["reply"], "Found")

    def test_scoring_context_has_no_required_rep_lookup(self):
        with patch.object(agent.gtm_agent, "invoke", return_value={
            "messages": [AIMessage(content="Scored")],
        }) as invoke:
            result = agent.run_agent("Score LEAD-12853 for OFFER-10001.", user_id=self.rep["rep_id"])
        context = agent.SYSTEM_PROMPT + invoke.call_args.args[0]["messages"][0]["content"]
        self.assertNotIn("get_current_rep", context)
        self.assertEqual(result["reply"], "Scored")
        self.assertEqual(invoke.call_args.kwargs["config"]["metadata"]["request_intent"], "score_prospect")

    def test_rep_lookup_tool_remains_available(self):
        self.assertIn(agent.get_current_rep, create_agent.call_args.kwargs["tools"])
        runtime = SimpleNamespace(config={"metadata": {"user_id": self.rep["rep_id"]}})
        self.assertEqual(agent.get_current_rep.func(runtime), {"rep": self.rep, "found": True})

    def test_email_sender_defaults_to_metadata_rep(self):
        runtime = SimpleNamespace(config={"metadata": {"user_id": self.rep["rep_id"]}})
        result = agent.send_prospect_email.func(
            {"name": "Prospect", "email": "prospect@example.com"},
            "Hello", "Hello from the team", runtime,
        )
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["from"], self.rep["email"])
        self.assertEqual(result["from_name"], self.rep["name"])


if __name__ == "__main__":
    unittest.main()
