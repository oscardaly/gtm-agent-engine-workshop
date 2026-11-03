import unittest
from types import SimpleNamespace
from unittest.mock import patch

from . import gtm_agent


class SendProspectEmailTests(unittest.TestCase):
    def send(self, prospect):
        return gtm_agent.send_prospect_email.func(
            prospect=prospect,
            subject="Introduction",
            body="Hello",
            runtime=SimpleNamespace(config={}),
            from_rep={"email": "rep@example.com", "name": "Rep"},
        )

    def test_disqualified_source_blocks_forged_eligible_contact(self):
        with patch.object(gtm_agent.data_service, "get_prospect_record", return_value={
            "disqualified": True, "email": "source@example.com", "name": "Source",
        }), patch.object(gtm_agent.uuid, "uuid4") as message_id:
            result = self.send({
                "prospect_id": "LEAD-1", "disqualified": False,
                "email": "forged@example.com", "name": "Forged",
            })
        self.assertEqual(result, {
            "status": "blocked", "error": "Prospect is disqualified; email not sent.",
        })
        message_id.assert_not_called()

    def test_unknown_or_missing_prospect_id_does_not_send(self):
        for prospect in ({"prospect_id": "UNKNOWN"}, {}, {"prospect_id": ""}):
            with self.subTest(prospect=prospect):
                with patch.object(gtm_agent.data_service, "get_prospect_record", return_value=None), \
                        patch.object(gtm_agent.uuid, "uuid4") as message_id:
                    result = self.send({**prospect, "email": "forged@example.com"})
                self.assertEqual(result, {
                    "status": "failed", "error": "Unknown prospect; email not sent.",
                })
                message_id.assert_not_called()

    def test_eligible_source_supplies_recipient_details(self):
        with patch.object(gtm_agent.data_service, "get_prospect_record", return_value={
            "disqualified": False, "email": "source@example.com", "name": "Source",
        }):
            result = self.send({
                "prospect_id": "LEAD-1", "email": "forged@example.com", "name": "Forged",
            })
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "source@example.com")
        self.assertEqual(result["to_name"], "Source")
        self.assertEqual(result["subject"], "Introduction")
        self.assertEqual(result["body"], "Hello")
        self.assertTrue(result["message_id"])

    def test_missing_source_email_does_not_use_supplied_email(self):
        with patch.object(gtm_agent.data_service, "get_prospect_record", return_value={
            "disqualified": False, "name": "Source",
        }), patch.object(gtm_agent.uuid, "uuid4") as message_id:
            result = self.send({"prospect_id": "LEAD-1", "email": "forged@example.com"})
        self.assertEqual(result, {
            "status": "failed", "error": "Prospect record has no email address.",
        })
        message_id.assert_not_called()
