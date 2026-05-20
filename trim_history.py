"""
Trim NAV history to last 5 years across all schemes.
Drops old NAVs to free Neon storage.
"""

import os
from datetime import date, timedelta
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from mutual_fund_data import db_cursor


def main():
    cutoff = date.today() - timedelta(days=5 * 365)
    print(f"Trimming NAV history older than {cutoff}")

    with db_cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM nav_history WHERE nav_date < %s", (cutoff,))
        to_delete = cur.fetchone()['n']
        cur.execute("SELECT COUNT(*) AS n FROM nav_history")
        total = cur.fetchone()['n']

    print(f"Total NAVs: {total}")
    print(f"Will delete: {to_delete} ({100*to_delete/total:.1f}%)")
    print(f"Will keep:   {total - to_delete}")

    confirm = input("\nProceed? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Aborted.")
        return

    print("Deleting old NAV rows in batches...")
    deleted = 0
    while True:
        with db_cursor() as cur:
            cur.execute(
                "DELETE FROM nav_history WHERE ctid = ANY (ARRAY("
                "  SELECT ctid FROM nav_history WHERE nav_date < %s LIMIT 50000"
                "))",
                (cutoff,)
            )
            n = cur.rowcount
        deleted += n
        print(f"  Deleted {deleted} so far...")
        if n == 0:
            break

    print(f"\nDone. Deleted {deleted} NAV rows.")
    print("Running VACUUM FULL to reclaim space...")

    import psycopg2
    url = os.environ['DATABASE_URL']
    if '?' in url:
        base = url.split('?')[0]
        url = base + '?sslmode=require'
    conn = psycopg2.connect(url)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("VACUUM FULL nav_history")
    conn.close()
    print("VACUUM complete.")

    print("\nUpdating backfill metadata...")
    with db_cursor() as cur:
        cur.execute("""
            UPDATE scheme_backfill_meta m SET
              oldest_nav_date = (SELECT MIN(nav_date) FROM nav_history WHERE scheme_code = m.scheme_code),
              nav_count       = (SELECT COUNT(*) FROM nav_history WHERE scheme_code = m.scheme_code)
        """)

    with db_cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM nav_history")
        print(f"\nFinal NAV count: {cur.fetchone()['n']}")


if __name__ == '__main__':
    main()
