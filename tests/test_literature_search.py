import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from openpyxl import Workbook, load_workbook

spec = importlib.util.spec_from_file_location('literature', Path(__file__).resolve().parents[1] / 'scripts/07_search_literature.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class LiteratureTests(unittest.TestCase):
    def test_source_and_export(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'source.xlsx'
            wb = Workbook()
            ws = wb.active
            ws.append(['EU export'])
            ws.append(['Active Substance ID', 'Substance', 'CAS Number', 'Status'])
            ws.append([1, 'Thiabendazole', '148-79-8', 'Not approved'])
            ws.append([2, 'Other', None, 'Approved'])
            wb.save(path)
            subs = m.substances(path)
            self.assertEqual(len(subs), 2)
            self.assertIn('148-79-8 Raman', m.queries(subs[0]))
            con = sqlite3.connect(':memory:')
            m.initialize(con)
            for provider, record in [('crossref', {'DOI': '10.1/ABC', 'title': ['=test']}),
                                     ('openalex', {'doi': 'https://doi.org/10.1/abc', 'title': '=test'})]:
                meta = m.normalize(record, provider)
                con.execute('INSERT INTO candidate VALUES (?,?,?,?,?,?)',
                            ('1', provider, 'test Raman', meta['key'], json.dumps(meta), '{}'))
            output = Path(d) / 'out.xlsx'
            m.export(con, subs, output)
            result = load_workbook(output)
            self.assertEqual(result['Candidates'].max_row, 2)
            self.assertEqual(result['Candidates']['E2'].data_type, 's')
            self.assertEqual(result['Candidates']['K2'].value, 'UNVERIFIED')
            self.assertEqual(result['Substances']['D3'].value, 0)
            con.close()

    def test_pagination_cap(self):
        old_get, old_sleep = m.get_json, m.time.sleep
        try:
            m.get_json = lambda *a: {'results': [{'id': 'work'}], 'meta': {'count': 20}}
            m.time.sleep = lambda _: None
            records, total, capped = m.search(None, 'openalex', 'test', 1, 'test@example.com')
            self.assertEqual(total, 20)
            self.assertTrue(capped)
            self.assertEqual(len(records), 1)
        finally:
            m.get_json, m.time.sleep = old_get, old_sleep


if __name__ == '__main__':
    unittest.main()
