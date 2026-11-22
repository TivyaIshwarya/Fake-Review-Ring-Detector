import sqlite3, sys
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

RELATIONS = {"rur": "net_rur", "rtr": "net_rtr", "rsr": "net_rsr"}

def pick_nodes(m, n, seed=42):
    """Keep nodes that have same-user edges (rings need them), fill rest randomly."""
    rng = np.random.default_rng(seed)
    total = m["label"].shape[1]
    rur = sp.coo_matrix(m["net_rur"])
    core = np.unique(np.concatenate([rur.row, rur.col]))
    if len(core) >= n:
        return np.sort(rng.choice(core, n, replace=False))
    rest = np.setdiff1d(np.arange(total), core)
    fill = rng.choice(rest, n - len(core), replace=False)
    return np.sort(np.concatenate([core, fill]))

def load(mat_path, db_path, n=15000):
    m = sio.loadmat(mat_path)
    keep = pick_nodes(m, n)
    new_id = {old: i for i, old in enumerate(keep)}
    labels = m["label"].ravel()[keep]
    feats = m["features"].tocsr()[keep].toarray()

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executescript("""
    DROP TABLE IF EXISTS reviews; DROP TABLE IF EXISTS edges;
    CREATE TABLE reviews(id INTEGER PRIMARY KEY, label INTEGER, features TEXT);
    CREATE TABLE edges(src INTEGER, dst INTEGER, relation TEXT);
    CREATE INDEX idx_edges ON edges(relation, src);
    CREATE TABLE IF NOT EXISTS flags(idempotency_key TEXT PRIMARY KEY,
        review_id INTEGER, ring_id TEXT, status TEXT, created_at TEXT);
    CREATE TABLE IF NOT EXISTS jobs(job_id TEXT PRIMARY KEY, state TEXT,
        payload TEXT, error TEXT, updated_at TEXT);
    CREATE TABLE IF NOT EXISTS checkpoints(run_id TEXT, step TEXT,
        state TEXT, created_at TEXT);
    """)
    cur.executemany("INSERT INTO reviews VALUES (?,?,?)",
        [(i, int(labels[i]), ",".join(f"{x:.5f}" for x in feats[i]))
         for i in range(len(keep))])

    for rel, key in RELATIONS.items():
        sub = sp.triu(m[key].tocsr()[keep][:, keep], k=1).tocoo()
        cur.executemany("INSERT INTO edges VALUES (?,?,?)",
            [(int(r), int(c), rel) for r, c in zip(sub.row, sub.col)])
        print(rel, "edges:", sub.nnz)
    con.commit(); con.close()
    print("reviews:", len(keep), "fraud:", int(labels.sum()))

if __name__ == "__main__":
    load(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 15000)
