import unittest
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
class ObservedDataTests(unittest.TestCase):
 def test_chronological_field_data(self):
  d=pd.read_csv(ROOT/'data/observed/processed/ballycanew_next_day_supervised.csv',parse_dates=['date'])
  self.assertEqual(len(d),2568)
  counts=d.split.value_counts()
  self.assertEqual(int(counts['train']),1838)
  self.assertEqual(int(counts['validation']),365)
  self.assertEqual(int(counts['test']),365)
  self.assertLess(d[d.split=='train'].date.max(),d[d.split=='validation'].date.min())
 def test_benchmark_separately_identified(self):
  self.assertTrue((ROOT/'data/benchmark/generated_station_daily_observations.csv').exists())
  self.assertTrue((ROOT/'data/observed/raw/Ballycanew_data.csv').exists())
if __name__=='__main__':unittest.main()
