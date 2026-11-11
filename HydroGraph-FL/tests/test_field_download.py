import json
import unittest
from scripts.download_field_data import parse_usgs_daily,parse_wqp_csv,join_exact_dates,daily_calendar

class FieldDataParsingTests(unittest.TestCase):
    def test_usgs_modern_geojson(self):
        data={'features':[{'properties':{'monitoring_location_id':'USGS-01570500','statistic_id':'00003','parameter_code':'00060','time':'2020-01-01','value':'350','unit_of_measure':'ft^3/s','approvals_status':['Approved']}}, {'properties':{'monitoring_location_id':'USGS-01576000','statistic_id':'00003','parameter_code':'00060','time':'2020-01-01','value':'21','unit_of_measure':'ft^3/s'}}]}
        f=parse_usgs_daily(data,'01570500')
        self.assertEqual(len(f),1)
        self.assertAlmostEqual(f.discharge_m3s.iloc[0],350*.028316846592)
    def test_discrete_and_no_imputation(self):
        raw=b'MonitoringLocationIdentifier,ActivityStartDate,CharacteristicName,ResultMeasureValue,ResultMeasure/MeasureUnitCode,ActivityIdentifier\nUSGS-01570500,2020-01-01,Turbidity,12.7,NTU,A1\nUSGS-01570500,2020-01-02,Turbidity,,NTU,A2\n'
        t=parse_wqp_csv(raw,'01570500')
        f=parse_usgs_daily({'features':[{'properties':{'monitoring_location_id':'USGS-01570500','statistic_id':'00003','parameter_code':'00060','time':'2020-01-01','value':'350','unit_of_measure':'ft^3/s'}}]},'01570500')
        self.assertEqual(len(t),1)
        self.assertEqual(len(join_exact_dates(f,t)),1)
        cal=daily_calendar(f,t,'2020-01-01','2020-01-03','01570500')
        self.assertEqual(len(cal),3)
        self.assertEqual(int(cal.turbidity_ntu_observed_sample_median.isna().sum()),2)
    def test_fnu_not_merged_with_ntu(self):
        raw=b'MonitoringLocationIdentifier,ActivityStartDate,CharacteristicName,ResultMeasureValue,ResultMeasure/MeasureUnitCode\nUSGS-01570500,2020-01-01,Turbidity,5,NTU\nUSGS-01570500,2020-01-01,Turbidity,6,FNU\n'
        t=parse_wqp_csv(raw,'01570500')
        self.assertEqual(set(t.turbidity_unit),{'NTU','FNU'})

if __name__=='__main__':unittest.main()
