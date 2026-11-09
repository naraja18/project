from pathlib import Path
import json, sqlite3
ROOT=Path(__file__).resolve().parents[1]
a=json.loads((ROOT/'outputs/part_a_audit.json').read_text())
checks=[a['raw_orders_rows']==13084,a['exact_duplicates']==119,a['pending_rows']==656,
 a['distinct_order_ids']==12309,a['clean_orders']==12309,a['completed']==11110,
 a['cancelled']==1199,a['completed_revenue']==2998668.80,a['missing_completed_delivery']==0,
 a['missing_cancelled_delivery']==1199,a['missing_clean_age']==0]
with sqlite3.connect(ROOT/'data/instamart.db') as con:
    tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
checks.append({'customers','products','orders'}.issubset(tables))
assert all(checks), 'One or more Part A reference checkpoints failed'
print('ALL CHECKPOINTS PASSED')
