import duckdb
from pathlib import Path

DB_PATH = Path(
    r"C:\Users\kjchr\Desktop\hockey\new_hockey_env\API data\duckdb\nhl_data.duckdb"
)

def run_sql_file(con, path):
    with open(path, "r", encoding="utf-8") as f:
        sql = f.read()
    con.execute(sql)

def main():
    con = duckdb.connect(DB_PATH)

    print("🔒 Applying DB indexes...")
    run_sql_file(con, DB_PATH.parent / "indexes.sql")

    print("✅ Database constraints applied.")

if __name__ == "__main__":
    main()
