"""Cover the UI's data joins, clarification decisions and model compatibility."""
from __future__ import annotations

import csv
from datetime import date
from unittest.mock import patch
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase

from .agents import Agents
from .app import app, explain, predict
from .catalog import Catalog, SUPPORTED_TIMES
from .conversation import clarify
from .schema import Conditions, Draft, ExplainRequest, ExplanationRefs, Intent, RoomSelection, ScoringRule
from .scoring import rank


class OfflineAgents(Agents):
    def __init__(self):
        self.intent_agent = self.room_agent = self.explanation_agent = self.translation_agent = None


class WorkspaceTests(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog()

    async def interpret(self, text, draft=None, language="en"):
        return await clarify(text, draft or Draft(), self.catalog, OfflineAgents(), date(2026, 11, 21), language)

    async def test_ready_and_ambiguous_inputs(self):
        reply = await self.interpret("BM223，6 月 15 日下午 2 点")
        self.assertEqual(reply["action"], "predict")
        self.assertEqual(reply["draft"], {"room_id": "BM223", "day": 166, "hour": 14, "preference": None})
        reply = await self.interpret("5楼下午，发电优先")
        self.assertEqual(reply["action"], "clarify")
        self.assertEqual(reply["draft"]["preference"], "power")
        self.assertEqual(reply["issues"][0]["floor"], 5)
        self.assertTrue(reply["issues"][0]["rooms"])
        self.assertEqual([(issue["field"], issue["reason"]) for issue in reply["issues"]],
                         [("room", "multiple"), ("day", "missing"), ("hour", "ambiguous")])

    async def test_default_english_and_bilingual_clarification(self):
        reply = await self.interpret("Predict BM223 for June 15 at 2 pm, balanced.")
        self.assertEqual(reply["language"], "en")
        self.assertEqual(reply["draft"], {"room_id": "BM223", "day": 166, "hour": 14, "preference": "balanced"})
        reply = await self.interpret("BM223, 15th of June at 2:00 PM.")
        self.assertEqual((reply["draft"]["day"], reply["draft"]["hour"]), (166, 14))
        reply = await self.interpret("Find a room on the 5th floor in the afternoon.")
        self.assertEqual(reply["issues"][0]["floor"], 5)
        self.assertEqual(reply["issues"][2]["reason"], "ambiguous")
        self.assertEqual((await self.interpret("5楼下午"))["language"], "en")

    async def test_explicit_language_only_or_combined_with_prediction(self):
        draft = Draft(room_id="BM223", day=166, hour=14)
        reply = await self.interpret("Could we talk in Chinese, please?", draft)
        self.assertEqual((reply["action"], reply["requested_language"]), ("language", "zh"))
        self.assertEqual(reply["draft"], draft.model_dump())
        reply = await self.interpret("Please reply in Chinese, and predict BM223 for June 15 at 2 pm.")
        self.assertEqual((reply["action"], reply["language"]), ("predict", "zh"))
        reply = await self.interpret("请用英文回答", draft, "zh")
        self.assertEqual((reply["action"], reply["language"]), ("language", "en"))
        reply = await self.interpret("六月十五日下午三点", draft, "zh")
        self.assertEqual((reply["language"], reply["draft"]["hour"]), ("zh", 15))
        reply = await self.interpret("Please respond in French.", draft)
        self.assertEqual((reply["action"], reply["language"]), ("language", "fr"))
        self.assertIsNone(reply["localized_reply"])

    async def test_local_explanations_use_the_requested_language(self):
        conditions = Conditions(room_id="BM223", day=1, hour=7)
        with patch.dict("os.environ", {"PVSD_LOCAL_ONLY": "1"}):
            async with app.router.lifespan_context(app):
                en = await explain(ExplainRequest(conditions=conditions))
                zh = await explain(ExplainRequest(conditions=conditions, language="zh"))
        self.assertIn("highest predicted PV energy", en["text"])
        self.assertIn("预测发电量最高", zh["text"])
        self.assertEqual((en["language"], zh["language"]), ("en", "zh"))

    async def test_conflicts_out_of_range_and_inheritance(self):
        draft = Draft(room_id="BM223", day=166, hour=14, preference="balanced")
        reply = await self.interpret("改到晚上7点", draft)
        self.assertEqual(reply["issues"], [{"field": "hour", "reason": "out_of_range", "options": []}])
        self.assertIsNone(reply["draft"]["hour"])
        reply = await self.interpret("改到下周末", draft)
        self.assertEqual(reply["issues"], [{"field": "day", "reason": "ambiguous", "options": []}])
        reply = await self.interpret("上午10点还是下午2点", draft)
        self.assertEqual(reply["issues"], [{"field": "hour", "reason": "multiple", "options": [10, 14]}])
        reply = await self.interpret("改到下午三点", draft)
        self.assertEqual(reply["draft"], {"room_id": "BM223", "day": 166, "hour": 15, "preference": "balanced"})
        self.assertEqual((await self.interpret("为什么推荐这个角度？", draft))["action"], "explain")
        self.assertEqual((await self.interpret("改到2月29日", draft))["issues"][0]["reason"], "out_of_range")

    async def test_sdk_typed_outputs_without_network(self):
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        agents = OfflineAgents()
        agents.intent_agent = Agent(TestModel(custom_output_args={"hours": [14]}), output_type=Intent)
        agents.room_agent = Agent(TestModel(custom_output_args={"room_ids": ["BM223", "UNKNOWN"]}),
                                  output_type=RoomSelection)
        agents.explanation_agent = Agent(TestModel(custom_output_args={"summary": "这是预测值。"}),
                                         output_type=ExplanationRefs)
        reply = await self.interpret("BM223，6 月 15 日下午 2 点")
        interpreted = await agents.interpret("下午2点", Draft(), "2026-11-21")
        self.assertEqual(interpreted.hours, [reply["draft"]["hour"]])
        self.assertEqual(await agents.match("房间", [{"id": "BM223"}]), ["BM223"])
        self.assertEqual(await agents.explain({}, "解释", "zh"), "这是预测值。")
        agents.translation_agent = Agent(TestModel(custom_output_args={"summary": "Je répondrai en français."}),
                                         output_type=ExplanationRefs)
        self.assertEqual(await agents.localize("I will reply in French.", "fr"), "Je répondrai en français.")

    async def test_model_matches_checked_in_sample(self):
        path = Path(__file__).resolve().parents[2] / "examples" / "trial_v0_2" / "expected_output.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            expected = list(csv.DictReader(stream))
        async with app.router.lifespan_context(app):
            for start in range(0, len(expected), 4):
                reference = expected[start:start + 4]
                condition = reference[0]
                prediction = predict(Conditions(room_id=condition["Room_ID"],
                                                day=int(condition["DOY_1based"]),
                                                hour=int(condition["Hour"])))
                self.assertEqual([row["PVSD_Angle_deg"] for row in prediction["rows"]], [0, 30, 60, 90])
                for actual, original in zip(prediction["rows"], reference):
                    for field, value in actual.items():
                        # Native float32 XGBoost inverse transforms vary by platform.
                        self.assertAlmostEqual(value, float(original[field]), delta=3e-7)

    def test_catalog_covers_the_simulation_clock(self):
        self.assertEqual(set(self.catalog.weather), SUPPORTED_TIMES)
        self.assertEqual(len(self.catalog.rooms), 476)
        self.assertEqual(self.catalog.inputs("BM223", 1, 7)["EPW_GHI_Wh_m2"], 6)


class RecommendationTests(TestCase):
    def test_backend_matches_frontend_scoring_fixture(self):
        angles = [0, 30, 60, 90]
        rows = [{"PVSD_Angle_deg": angle, "PV_AC_kWh": power, "F_UDI_fraction": udi,
                 "F_300_fraction": 0, "E_P10_lux": 0}
                for angle, power, udi in zip(angles, [.008, .006, .003, .001], [.4, .64, .8, .6])]
        for weight, angle in [(100, 0), (50, 30), (0, 60)]:
            rule = ScoringRule(power=weight, udi=50, f300=30, p10=20, p10_target=100)
            self.assertEqual([row["PVSD_Angle_deg"] for row in rank(rows, rule) if row["recommended"]], [angle])
