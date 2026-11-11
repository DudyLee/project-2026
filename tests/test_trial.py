"""Check released estimators without fitting or accessing the research dataset."""
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pvsd_trial import TrialPredictor, TARGETS
from pvsd_trial_v0_2 import ROOM_FEATURES, read_conditions


class TrialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guards = ExitStack()
        for model in (HistGradientBoostingRegressor, RandomForestRegressor, XGBRegressor):
            cls.guards.enter_context(patch.object(model, 'fit', side_effect=AssertionError('No training allowed')))
        cls.addClassCleanup(cls.guards.close)
        cls.current = TrialPredictor()
        cls.inputs = pd.read_csv(ROOT / 'examples/trial_v0_2/input.csv')
        cls.expected = pd.read_csv(ROOT / 'examples/trial_v0_2/expected_output.csv')

    def assert_predictions(self, actual, expected):
        np.testing.assert_allclose(actual[TARGETS], expected[TARGETS], rtol=1e-10, atol=1e-8)
        np.testing.assert_array_equal(actual.PVSD_Angle_deg, expected.PVSD_Angle_deg)

    def test_default_and_frozen_predictions(self):
        result = self.current.predict(self.inputs)
        self.assertEqual(self.current.version, '0.2.0')
        self.assertEqual(len(result), 4 * len(self.inputs))
        self.assert_predictions(result, self.expected)

    def test_json_and_batch(self):
        records = read_conditions(ROOT / 'examples/trial_v0_2/input.json')
        self.assert_predictions(self.current.predict(records, batch_size=1), self.expected)

    def test_bim_lookup(self):
        with tempfile.TemporaryDirectory(prefix='.trial-test-', dir=ROOT) as temporary:
            path = Path(temporary) / 'bim.csv'
            self.inputs[['Room_ID'] + ROOM_FEATURES].drop_duplicates().to_csv(path, index=False)
            model = TrialPredictor(bim_path=path)
            query = self.inputs.drop(columns=ROOM_FEATURES)
            self.assert_predictions(model.predict(query), self.expected)
            query = query.copy()
            query['Room_Area_m2'] = self.inputs.Room_Area_m2 + 1
            with self.assertRaises(ValueError):
                model.predict(query)

    def test_invalid_inputs(self):
        bad = [self.inputs.drop(columns='Annual_Incident_Shaded_kWh_m2'),
               self.inputs.assign(E_mean_lux=100), self.inputs.assign(Hour=3),
               self.inputs.assign(PVSD_Angle_deg=30), self.inputs.assign(Glass_Area_m2=0)]
        for frame in bad:
            with self.subTest(columns=list(frame.columns)):
                with self.assertRaises(ValueError):
                    self.current.predict(frame)

    def test_legacy_rollback(self):
        model = TrialPredictor(version='0.1')
        inputs = pd.read_csv(ROOT / 'examples/trial_v0_1/input.csv')
        expected = pd.read_csv(ROOT / 'examples/trial_v0_1/expected_output.csv')
        self.assert_predictions(model.predict(inputs), expected)

    def test_cli_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory(prefix='.trial-test-', dir=ROOT) as temporary:
            output = Path(temporary) / 'predictions.csv'
            command = [sys.executable, '-m', 'pvsd_trial', '--input',
                       'examples/trial_v0_2/input.csv', '--output', str(output)]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assert_predictions(pd.read_csv(output), self.expected)
            repeated = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(repeated.returncode, 2)
            self.assertIn('Output exists', repeated.stderr)


if __name__ == '__main__':
    unittest.main()
