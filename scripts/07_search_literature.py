"""Resumable metadata discovery; candidates are NOT confirmed experimental spectra."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from datetime import datetime, timezone

import requests
from openpyxl import load_workbook, Workbook

ROOT = Path(__file__).resolve().parents[1]
TERMS = ('Raman', 'SERS', 'surface enhanced Raman')


def substances(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = iter(wb.active.values)
        for row in rows:
            headers = [str(x or '').strip() for x in row]
            if 'Active Substance ID' in headers:
                break
        else:
            raise ValueError('Missing Active Substance ID header')
        for col in ('Substance', 'CAS Number'):
            if col not in headers:
                raise ValueError('Missing column: ' + col)
        result = []
        for row in rows:
            r = dict(zip(headers, row))
            if r.get('Active Substance ID') is not None and r.get('Substance'):
                result.append({'id': str(r['Active Substance ID']),
                               'name': str(r['Substance']).strip(),
                               'cas': str(r.get('CAS Number') or '').strip()})
        return result
    finally:
        wb.close()


def queries(s):
    identifiers = [s['name']] + re.findall(r'\b\d{2,7}-\d{2}-\d\b', s['cas'])
    return sorted({f'{name} {term}' for name in identifiers for term in TERMS})


def get_json(session, url, params, headers):
    for attempt in range(5):
        try:
            response = session.get(url, params=params, headers=headers, timeout=45)
        except requests.RequestException:
            if attempt == 4:
                raise RuntimeError('Network request failed (URL omitted to protect credentials)') from None
            time.sleep(2 ** attempt)
            continue
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 4:
                raise RuntimeError(f'HTTP {response.status_code}; retry on next run')
            try:
                delay = float(response.headers.get('Retry-After', 2 ** attempt))
            except ValueError:
                delay = 2 ** attempt
            if delay > 60:
                raise RuntimeError('Provider requests a long pause; rerun later')
            time.sleep(max(1, delay))
            continue
        if not response.ok:
            raise RuntimeError(f'HTTP {response.status_code}; check provider access/key')
        return response.json()


def search(session, provider, query, pages, email):
    records = []
    total = 0
    for page in range(1, pages + 1):
        headers = {'User-Agent': f'Water4All-literature-discovery/1.0 (mailto:{email})'}
        if provider == 'openalex':
            key = os.getenv('OPENALEX_API_KEY')
            if key:
                headers['Authorization'] = 'Bearer ' + key
            data = get_json(session, 'https://api.openalex.org/works',
                            {'search': query, 'per_page': 100, 'page': page}, headers)
            items, total = data['results'], data['meta']['count']
        else:
            data = get_json(session, 'https://api.crossref.org/works',
                            {'query.bibliographic': query, 'rows': 100,
                             'offset': (page - 1) * 100, 'mailto': email}, headers)['message']
            items, total = data['items'], data['total-results']
        records.extend(items)
        time.sleep(1)
        if not items or len(records) >= total:
            break
    return records, total, len(records) < total


def normalize(item, provider):
    doi = str(item.get('doi') or item.get('DOI') or '').lower().strip()
    doi = re.sub(r'^https?://(?:dx\.)?doi.org/', '', doi)
    title = item.get('title') or item.get('display_name') or ''
    if isinstance(title, list):
        title = ' '.join(title)
    loc = item.get('best_oa_location') or {}
    licenses = item.get('license') or []
    license_text = loc.get('license') or ''
    if provider == 'crossref':
        license_text = '; '.join(x.get('URL', '') for x in licenses)
    return {'key': doi or item.get('id') or hashlib.sha256(title.encode()).hexdigest(),
            'doi': doi, 'title': title,
            'url': ('https://doi.org/' + doi) if doi else item.get('id', ''),
            'oa_url': loc.get('pdf_url') or loc.get('landing_page_url') or '',
            'license_metadata': license_text}


def initialize(con):
    con.executescript('''
        CREATE TABLE IF NOT EXISTS query_log (
          substance_id TEXT, provider TEXT, query TEXT, pages INTEGER,
          status TEXT, total INTEGER, capped INTEGER, checked_at TEXT, error TEXT,
          PRIMARY KEY(substance_id, provider, query, pages));
        CREATE TABLE IF NOT EXISTS candidate (
          substance_id TEXT, provider TEXT, query TEXT, work_key TEXT,
          metadata TEXT, raw_json TEXT,
          PRIMARY KEY(substance_id, provider, query, work_key));
    ''')


def export(con, subs, output):
    wb = Workbook()
    ws = wb.active
    ws.title = 'Candidates'
    ws.append(['substance_id', 'name', 'cas', 'doi', 'title', 'url', 'oa_url',
               'license_metadata', 'providers', 'queries', 'spectrum_status',
               'technique_confirmed', 'figure', 'supplementary_url', 'notes'])
    combined = {}
    for sid, provider, query, key, meta in con.execute(
            'SELECT substance_id,provider,query,work_key,metadata FROM candidate'):
        entry = combined.setdefault((sid, key), {'meta': json.loads(meta), 'providers': set(), 'queries': set()})
        entry['providers'].add(provider)
        entry['queries'].add(query)
        for k, v in json.loads(meta).items():
            if v and not entry['meta'].get(k):
                entry['meta'][k] = v
    lookup = {s['id']: s for s in subs}
    counts = {}
    def append(sheet, values):
        # Force strings to text, including titles beginning with '='.
        sheet.append([v[:32000] if isinstance(v, str) else v for v in values])
        for cell in sheet[sheet.max_row]:
            if isinstance(cell.value, str):
                cell.data_type = 's'
    for (sid, key), e in sorted(combined.items()):
        s = lookup.get(sid, {'name': '', 'cas': ''})
        m = e['meta']
        counts[sid] = counts.get(sid, 0) + 1
        append(ws, [sid, s['name'], s['cas']] + [m[k] for k in
               ('doi', 'title', 'url', 'oa_url', 'license_metadata')] +
               ['; '.join(sorted(e['providers'])), '; '.join(sorted(e['queries'])),
                'UNVERIFIED', '', '', '', ''])
    summary = wb.create_sheet('Substances')
    summary.append(['id', 'name', 'cas', 'candidate_count', 'interpretation'])
    for s in subs:
        append(summary, [s['id'], s['name'], s['cas'], counts.get(s['id'], 0),
                         'See Query_log; zero candidates does not prove absence of spectra'])
    log = wb.create_sheet('Query_log')
    log.append(['substance_id', 'provider', 'query', 'pages', 'status', 'total', 'capped', 'checked_at', 'error'])
    for r in con.execute('SELECT * FROM query_log ORDER BY substance_id,provider,query'):
        append(log, r)
    for sheet in wb:
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
    # Export is generated, not a manual-review database. Keep edits in a copy.
    temp = output.with_suffix('.tmp.xlsx')
    wb.save(temp)
    temp.replace(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xlsx', type=Path, default=ROOT / 'scripts/ActiveSubstanceExport_27-02-2026.xlsx')
    p.add_argument('--out', type=Path, default=ROOT / 'data/literature')
    p.add_argument('--email', default=os.getenv('CROSSREF_MAILTO', ''))
    p.add_argument('--providers', nargs='+', choices=['openalex', 'crossref'], default=['openalex', 'crossref'])
    p.add_argument('--limit', type=int, help='First N substances; omitted = all')
    p.add_argument('--pages', type=int, default=1, help='100 results/page/query; cap is logged')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--export-only', action='store_true')
    a = p.parse_args()
    if not 1 <= a.pages <= 100 or (a.limit is not None and a.limit < 1):
        p.error('pages must be 1..100; limit must be positive')
    all_subs = substances(a.xlsx)
    selected = all_subs[:a.limit]
    print(f'Substances: {len(selected)} / {len(all_subs)}; no regulatory-status filtering')
    if a.dry_run:
        print(f'Queries: {sum(len(queries(s)) for s in selected) * len(a.providers)}')
        return
    if not a.email and not a.export_only:
        p.error('Provide --email or CROSSREF_MAILTO for provider contact')
    a.out.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(a.out / 'literature.sqlite') as con, requests.Session() as session:
        initialize(con)
        try:
            if not a.export_only:
                for s in selected:
                    print(f"Searching {s['id']}: {s['name']}", flush=True)
                    for provider in a.providers:
                        for query in queries(s):
                            identity = (s['id'], provider, query, a.pages)
                            prior = con.execute('SELECT status FROM query_log WHERE substance_id=? AND provider=? AND query=? AND pages=?', identity).fetchone()
                            if prior and prior[0] == 'done':
                                continue
                            stamp = datetime.now(timezone.utc).isoformat()
                            try:
                                items, total, capped = search(session, provider, query, a.pages, a.email)
                                with con:
                                    for item in items:
                                        m = normalize(item, provider)
                                        con.execute('INSERT OR REPLACE INTO candidate VALUES (?,?,?,?,?,?)',
                                            (s['id'], provider, query, m['key'], json.dumps(m), json.dumps(item)))
                                    con.execute('INSERT OR REPLACE INTO query_log VALUES (?,?,?,?,?,?,?,?,?)',
                                                identity + ('done', total, int(capped), stamp, ''))
                            except (RuntimeError, ValueError, KeyError) as exc:
                                with con:
                                    con.execute('INSERT OR REPLACE INTO query_log VALUES (?,?,?,?,?,?,?,?,?)',
                                                identity + ('error', None, None, stamp, str(exc)))
                                raise RuntimeError(f'{provider}: search interrupted; progress saved. {exc}') from None
        finally:
            export(con, all_subs, a.out / 'literature_candidates.xlsx')
    print(f"Saved {a.out / 'literature_candidates.xlsx'}")


if __name__ == '__main__':
    main()
