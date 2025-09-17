
import argparse
import json
import os
import sqlite3
import pandas as pd
from tqdm import tqdm

def norm(s: str) -> str:
    return str(s).strip().lower() if s is not None else ""

SCHEMA = """
CREATE TABLE IF NOT EXISTS kg (
    term TEXT,
    term_cui TEXT,
    rel TEXT,
    rel_detail TEXT,
    related_term TEXT,
    related_cui TEXT
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_term ON kg(term)",
    "CREATE INDEX IF NOT EXISTS idx_related_term ON kg(related_term)",
    "CREATE INDEX IF NOT EXISTS idx_rel ON kg(rel)",
]

def insert_rows(conn, rows):
    conn.executemany(
        "INSERT INTO kg(term, term_cui, rel, rel_detail, related_term, related_cui) VALUES (?,?,?,?,?,?)",
        rows,
    )

def load_rel_csv(path: str):
    if not path or not os.path.exists(path):
        return []
    sep = "\t" if path.endswith(".tsv") else ","
    df = pd.read_csv(path, dtype=str, keep_default_na=False, sep=sep)
    cols = {c.lower(): c for c in df.columns}
    term = cols.get("term_name") or cols.get("term") or list(cols.values())[0]
    term_cui = cols.get("term_cui") or cols.get("cui")
    rel = cols.get("relationship") or cols.get("rel")
    rel_detail = cols.get("relationship_detail") or cols.get("detail")
    related_term = cols.get("related_term") or cols.get("related")
    related_cui = cols.get("related_cui")
    out = []
    for _, r in df.iterrows():
        out.append(
            (
                norm(r.get(term)),
                norm(r.get(term_cui)),
                norm(r.get(rel)),
                norm(r.get(rel_detail)),
                norm(r.get(related_term)),
                norm(r.get(related_cui)),
            )
        )
    return out

def stream_big_csv(path: str, chunk_size: int = 100_000):
    if not path or not os.path.exists(path):
        return []
    sep = "\t" if path.endswith(".tsv") else ","
    for chunk in pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=chunk_size, sep=sep):
        cols = {c.lower(): c for c in chunk.columns}
        term = cols.get("term_name") or cols.get("term") or list(cols.values())[0]
        term_cui = cols.get("term_cui") or cols.get("cui")
        rel = cols.get("relationship") or cols.get("rel")
        rel_detail = cols.get("relationship_detail") or cols.get("detail")
        related_term = cols.get("related_term") or cols.get("related")
        related_cui = cols.get("related_cui")
        rows = []
        for _, r in chunk.iterrows():
            rows.append(
                (
                    norm(r.get(term)),
                    norm(r.get(term_cui)),
                    norm(r.get(rel)),
                    norm(r.get(rel_detail)),
                    norm(r.get(related_term)),
                    norm(r.get(related_cui)),
                )
            )
        yield rows

def write_phonetic_jsonl(in_csv: str, out_jsonl: str):
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        if not in_csv or not os.path.exists(in_csv):
            return
        sep = "\t" if in_csv.endswith(".tsv") else ","
        df = pd.read_csv(in_csv, dtype=str, keep_default_na=False, sep=sep)
        cols = {c.lower(): c for c in df.columns}
        term_col = cols.get("medical_term") or cols.get("term") or list(cols.values())[0]
        sim_col = cols.get("similar_word") or cols.get("similar")
        cui_col = cols.get("cui")
        for _, r in df.iterrows():
            rec = {
                "term": norm(r.get(term_col)),
                "similar": norm(r.get(sim_col)),
                "cui": norm(r.get(cui_col)),
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phonetic_csv", required=True)
    ap.add_argument("--rel_csv", required=False)
    ap.add_argument("--rel_csv2", required=False)
    ap.add_argument("--kg_big_csv", required=False)
    ap.add_argument("--out_sqlite", required=True)
    ap.add_argument("--out_phonetic", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_sqlite), exist_ok=True)

    write_phonetic_jsonl(args.phonetic_csv, args.out_phonetic)

    conn = sqlite3.connect(args.out_sqlite)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(SCHEMA)

    for path in [args.rel_csv, args.rel_csv2]:
        rows = load_rel_csv(path) if path else []
        if rows:
            insert_rows(conn, rows)
            conn.commit()

    if args.kg_big_csv and os.path.exists(args.kg_big_csv):
        for rows in tqdm(stream_big_csv(args.kg_big_csv), desc="Streaming KG CSV"):
            if rows:
                insert_rows(conn, rows)
                conn.commit()

    for idx_sql in INDEXES:
        conn.execute(idx_sql)
    conn.commit()
    conn.close()

    print("KG build complete:")
    print(f"  SQLite: {args.out_sqlite}")
    print(f"  Phonetic JSONL: {args.out_phonetic}")

if __name__ == "__main__":
    main()
