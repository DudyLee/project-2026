"""Cover the UI's data joins, clarification decisions and model compatibility."""
from __future__ import annotations

import csv
from datetime import date
from unittest.mock import patch
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase

from .agents import Agents
from .app import app, explain, health, predict
from .catalog import Catalog, SUPPORTED_TIMES
from .conversation import clarify
from .schema import Conditions, Draft, ExplainRequest, ExplanationRefs, Intent, ResultRef, RoomSelection, ScoringRule
from .scoring import rank


class OfflineAgents(Agents):
    def __init__(self):
        self.intent_agent = self.room_agent = self.explanation_agent = self.translation_agent = None


class WorkspaceTests(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog()

    async def interpret(self, text, draft=None, language="en", results=None, active=None):
        return await clarify(text, draft or Draft(), self.catalog, OfflineAgents(), date(2026, 11, 21),
                             language, results, active)

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
                focused = await explain(ExplainRequest(conditions=conditions, angles=[60, 0], metrics=["E_P10_lux"]))
                model = health()
        self.assertEqual(model["model_version"], "0.2.0")
        self.assertEqual(model["weather_source"], "original_simulation_weather")
        self.assertIn("change from 60° to 0°", focused["text"])
        self.assertNotIn("PV energy", focused["text"])
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

    async def test_followups_use_result_references(self):
        draft = Draft(room_id="BM223", day=166, hour=14, preference="balanced")
        cards = [ResultRef(id="morning", room_id="BM223", day=166, hour=10),
                 ResultRef(id="afternoon", room_id="BM223", day=166, hour=14)]

        async def ask(message):
            return await self.interpret(message, draft, results=cards, active="afternoon")

        explain = await ask("Explain BM223")
        self.assertEqual((explain["action"], explain["result_choices"]), ("explain", ["morning", "afternoon"]))
        explain = await ask("Explain the first prediction")
        self.assertEqual((explain["action"], explain["result_id"]), ("explain", "morning"))
        compare = await ask("Compare with the previous result")
        self.assertEqual((compare["action"], compare["against_result_id"]), ("compare", "morning"))
        compare = await ask("Compare 0° and 60°")
        self.assertEqual((compare["action"], compare["angles"]), ("compare", [0, 60]))
        compare = await ask("Compare this with 3 pm")
        self.assertEqual((compare["action"], compare["compare_scenario"], compare["draft"]["hour"]),
                         ("predict", True, 15))
        compare = await ask("Compare this with tomorrow afternoon")
        self.assertEqual((compare["action"], compare["compare_scenario"], compare["issues"][0]["field"]),
                         ("clarify", True, "hour"))
        self.assertTrue((await ask("Explain BM224"))["reference_missed"])

    async def test_metric_scopes_survive_comparison_and_explanation(self):
        result = ResultRef(id="card", room_id="BM223", day=166, hour=14)
        async def ask(message):
            return await self.interpret(message, results=[result], active=result.id)

        daylight = await ask("Compare 0° and 60° on day light")
        self.assertEqual((daylight["action"], daylight["angles"]), ("compare", [0, 60]))
        self.assertEqual(daylight["metrics"], ["E_mean_lux", "E_P10_lux", "F_300_fraction", "F_UDI_fraction"])
        scenario = await ask("Compare with tomorrow afternoon for useful daylight")
        self.assertEqual((scenario["action"], scenario["compare_scenario"], scenario["metrics"]),
                         ("clarify", True, ["F_UDI_fraction"]))
        explanation = await ask("Why is P10 lower at 60° than 0°?")
        self.assertEqual((explanation["action"], explanation["angles"], explanation["metrics"]),
                         ("explain", [60, 0], ["E_P10_lux"]))
        self.assertEqual((await ask("Compare 0° and 60°"))["metrics"], [])

    async def test_ranking_preserves_objective_and_thresholds_in_model_units(self):
        result = ResultRef(id="card", room_id="BM223", day=166, hour=14)
        async def ask(message):
            return await self.interpret(message, results=[result], active=result.id)

        question = await ask("Which angle generates the most PV with at least 40% coverage at 300 lux?")
        self.assertEqual((question["action"], question["result_id"]), ("rank", result.id))
        self.assertEqual(question["filters"], [{"metric": "F_300_fraction", "operator": "gte", "value": .4}])
        self.assertEqual(question["sort"], {"metric": "PV_AC_kWh", "direction": "desc"})
        multi = await ask("Rank by lowest PV energy with P10 > 20 lux and PV energy at least 30 Wh.")
        self.assertEqual(multi["sort"]["direction"], "asc")
        self.assertEqual(multi["filters"], [
            {"metric": "E_P10_lux", "operator": "gt", "value": 20},
            {"metric": "PV_AC_kWh", "operator": "gte", "value": .03},
        ])
        self.assertEqual((await ask("Rank by PV energy with 40% coverage at 500 lux or P10 above 30 lux"))["ranking_issue"],
                         "filter")
        self.assertEqual((await ask("Rank PV with at least 40% coverage at 500 lux"))["ranking_issue"], "filter")
        chinese = await ask("哪个角度发电量最高，300 lux 达标比例至少 40%？")
        self.assertEqual(chinese["filters"], question["filters"])
        self.assertEqual(chinese["sort"], question["sort"])

    async def test_combined_daylight_rule_opens_with_weights_targets_and_cutoffs(self):
        result = ResultRef(id="card", room_id="BM223", day=166, hour=14)

        async def ask(message):
            return await self.interpret(message, results=[result], active=result.id)

        rule = await ask("Use 70% useful daylight and 30% coverage at 300 lux. Require P10 of at least 80 lux.")
        self.assertEqual((rule["action"], rule["result_id"]), ("rule", result.id))
        self.assertEqual(rule["rule"], {
            "udi": 70, "f300": 30,
            "cutoffs": [{"metric": "E_P10_lux", "operator": "gte", "value": 80}],
        })
        four = await ask("Use 40% useful daylight, 30% coverage at 300 lux, 20% P10 with full credit at 100 lux, and 10% mean illuminance with full credit at 500 lux.")
        self.assertEqual(four["rule"], {"udi": 40, "f300": 30, "p10": 20, "p10_target": 100,
                                        "mean": 10, "mean_target": 500})
        self.assertEqual((await ask("Remove the P10 cutoff."))["rule"], {"remove_cutoffs": ["E_P10_lux"]})
        self.assertEqual((await ask("Require P10 of at least 80%."))["rule_issue"], "cutoff")
        self.assertEqual((await ask("要求 P10 至少 80 lux。"))["rule"]["cutoffs"], rule["rule"]["cutoffs"])

    async def test_scoring_chart_export_and_help_do_not_predict(self):
        result = ResultRef(id="card", room_id="BM223", day=166, hour=14)

        async def ask(message, language="en"):
            return await self.interpret(message, language=language, results=[result], active=result.id)

        for query in ("Make PV 70% and take P10 out", "光伏占70%，去掉P10"):
            rule = await ask(query)
            self.assertEqual((rule["action"], rule["rule"]), ("rule", {"power": 70, "p10": 0}))
        chart = await ask("Show only PV energy and UDI")
        self.assertEqual((chart["action"], chart["visibility"], chart["metrics"]),
                         ("chart", "only", ["F_UDI_fraction", "PV_AC_kWh"]))
        self.assertEqual(len((await ask("Restore all metrics"))["metrics"]), 5)
        self.assertEqual((await ask("Highlight 60°"))["angles"], [60])
        self.assertTrue((await ask("Open the values table"))["table_open"])
        export = await ask("Export 30° and 60° as JSON")
        self.assertEqual((export["action"], export["format"], export["angles"]), ("export", "json", [30, 60]))
        self.assertEqual((await ask("Export 15° as CSV"))["unsupported_angles"], [15])
        self.assertIn("10th percentile", (await ask("What is P10?"))["localized_reply"])
        self.assertIn("第 10 百分位", (await ask("什么是P10？", "zh"))["localized_reply"])
        self.assertEqual((await ask("What meetings do I have at 2 pm?"))["action"], "help")

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
        agents.intent_agent = Agent(TestModel(custom_output_args={"action": "chart", "angles": [60],
                                                                  "result_id": "unknown"}),
                                    output_type=Intent)
        ref = ResultRef(id="active", room_id="BM223", day=166, hour=14)
        routed = await clarify("Please focus there.", Draft(), self.catalog, agents, date(2026, 11, 21),
                               results=[ref], active=ref.id)
        self.assertEqual((routed["action"], routed["result_id"], routed["angles"]), ("chart", "active", [60]))
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
    def test_preferences_apply_lux_targets_and_cutoffs_before_ranking(self):
        rule = ScoringRule(power=50, udi=40, f300=30, p10=20, p10_target=100,
                           mean=10, mean_target=500,
                           cutoffs=[{"metric": "E_P10_lux", "operator": "gte", "value": 80}])
        rows = [
            {"PVSD_Angle_deg": 0, "PV_AC_kWh": .008, "F_UDI_fraction": .8, "F_300_fraction": .9,
             "E_P10_lux": 60, "E_mean_lux": 550},
            {"PVSD_Angle_deg": 30, "PV_AC_kWh": .006, "F_UDI_fraction": .7, "F_300_fraction": .5,
             "E_P10_lux": 160, "E_mean_lux": 250},
        ]
        scored = rank(rows, rule)
        self.assertEqual([row["daylight_score"] for row in scored], [81, 68])
        self.assertEqual([row["preference_score"] for row in scored], [None, 100])
        self.assertEqual([row["PVSD_Angle_deg"] for row in scored if row["recommended"]], [30])
        self.assertFalse(any(row["recommended"] for row in rank(rows[:1], rule)))

    def test_backend_matches_frontend_scoring_fixture(self):
        angles = [0, 30, 60, 90]
        rows = [{"PVSD_Angle_deg": angle, "PV_AC_kWh": power, "F_UDI_fraction": udi,
                 "F_300_fraction": 0, "E_P10_lux": 0}
                for angle, power, udi in zip(angles, [.008, .006, .003, .001], [.4, .64, .8, .6])]
        for weight, angle in [(100, 0), (50, 30), (0, 60)]:
            rule = ScoringRule(power=weight, udi=50, f300=30, p10=20, p10_target=100)
            self.assertEqual([row["PVSD_Angle_deg"] for row in rank(rows, rule) if row["recommended"]], [angle])
