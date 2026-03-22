import os
import unittest
from importlib.machinery import SourceFileLoader
from unittest.mock import patch


selfrag = SourceFileLoader("selfrag", "SELFRAG-agent.py").load_module()


class FakeGraph:
    def invoke(self, payload):
        return {
            "question": payload.get("question"),
            "generation": "mocked answer",
            "hallucinated": False,
            "valid_answer": True,
        }


class TestSelfRAGSmoke(unittest.TestCase):
    def test_create_model_requires_openai_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with self.assertRaises(ValueError):
                selfrag.create_model({})

    def test_run_self_rag_uses_compiled_graph(self):
        with patch.object(selfrag, "build_graph", return_value=FakeGraph()):
            response = selfrag.run_self_rag("what is rag?")

        self.assertEqual(response["question"], "what is rag?")
        self.assertEqual(response["generation"], "mocked answer")
        self.assertFalse(response["hallucinated"])
        self.assertTrue(response["valid_answer"])

    def test_print_response_summary(self):
        response = {
            "generation": "ok",
            "hallucinated": False,
            "valid_answer": True,
        }
        selfrag.print_response_summary(response)


if __name__ == "__main__":
    unittest.main()
