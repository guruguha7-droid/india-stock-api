"""
Universe cleanup: trim to Growth + Direct plans only.

Drops:
- IDCW / Dividend variants (same fund, different payout mechanism)
- Regular plans (same fund, higher expense ratio)
- Bonus / special variants
- FMP (Fixed Maturity Plans) and other closed-ended schemes
"""

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from mutual_fund_data import db_cursor


def is_keeper(scheme_name):
    """Return True if this scheme should be kept (Growth + Direct only)."""
    name = scheme_name.upper()

    # Must be Direct plan
    if 'DIRECT' not in name:
        return False

    # Must be Growth (not dividend/IDCW)
    if 'GROWTH' not in name:
        return False

    # Drop dividend payout variants even if labeled Growth+Direct
    drop_keywords = [
        'IDCW', 'DIVIDEND', 'BONUS',
        'WEEKLY', 'MONTHLY', 'QUARTERLY', 'DAILY',
        'PAYOUT', 'REINVESTMENT',
        'FMP', 'FIXED MATURITY',  # closed-ended
    ]
    for kw in drop_keywords:
        if kw in name:
            return False

    return True


def main():
    with db_cursor() as cur:
        cur.execute("SELECT scheme_code, scheme_name FROM schemes")
        all_schemes = cur.fetchall()

    print(f"Total schemes in DB: {len(all_schemes)}")

    keepers = []
    drops = []
    for r in all_schemes:
        if is_keeper(r['scheme_name']):
            keepers.append(r['scheme_code'])
        else:
            drops.append(r['scheme_code'])

    print(f"\nKeeping: {len(keepers)} schemes (Growth + Direct only)")
    print(f"Dropping: {len(drops)} schemes (variants, dividend, regular, etc.)")

    print(f"\nSample kept:")
    with db_cursor() as cur:
        cur.execute("SELECT scheme_name FROM schemes WHERE scheme_code = ANY(%s) LIMIT 5",
                    (keepers[:5],))
        for r in cur.fetchall():
            print(f"  KEEP: {r['scheme_name']}")

    print(f"\nSample dropped:")
    with db_cursor() as cur:
        cur.execute("SELECT scheme_name FROM schemes WHERE scheme_code = ANY(%s) LIMIT 5",
                    (drops[:5],))
        for r in cur.fetchall():
            print(f"  DROP: {r['scheme_name']}")

    confirm = input(f"\nProceed with deletion of {len(drops)} schemes? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Aborted.")
        return

    print(f"\nDeleting NAV history for {len(drops)} dropped schemes...")
    batch_size = 500
    deleted_navs = 0
    deleted_meta = 0
    deleted_schemes = 0

    for i in range(0, len(drops), batch_size):
        batch = drops[i:i+batch_size]
        with db_cursor() as cur:
            cur.execute("DELETE FROM nav_history WHERE scheme_code = ANY(%s)", (batch,))
            deleted_navs += cur.rowcount

            cur.execute("DELETE FROM scheme_backfill_meta WHERE scheme_code = ANY(%s)", (batch,))
            deleted_meta += cur.rowcount

            cur.execute("DELETE FROM schemes WHERE scheme_code = ANY(%s)", (batch,))
            deleted_schemes += cur.rowcount

        print(f"  Batch {i//batch_size + 1}: deleted {len(batch)} schemes, "
              f"running total {deleted_navs} NAVs, {deleted_schemes} schemes")

    print(f"\nDone:")
    print(f"  Deleted {deleted_schemes} schemes")
    print(f"  Deleted {deleted_navs} NAV rows")
    print(f"  Deleted {deleted_meta} backfill_meta rows")

    print("\nReclaiming space via VACUUM FULL...")
    import psycopg2
    conn = psycopg2.connect(os.environ['DATABASE_URL'].split('?')[0] + '?sslmode=require')
    conn.autocommit = True  # VACUUM FULL can't run inside a transaction
    cur = conn.cursor()
    cur.execute("VACUUM FULL nav_history")
    cur.execute("VACUUM FULL schemes")
    cur.execute("VACUUM FULL scheme_backfill_meta")
    conn.close()
    print("VACUUM complete.")

    with db_cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM schemes")
        scheme_count = cur.fetchone()['n']
        cur.execute("SELECT COUNT(*) AS n FROM nav_history")
        nav_count = cur.fetchone()['n']

    print(f"\nFinal: {scheme_count} schemes | {nav_count} NAV rows")


if __name__ == '__main__':
    main()
