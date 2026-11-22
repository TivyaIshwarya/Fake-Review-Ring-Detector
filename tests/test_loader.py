import sqlite3, pytest, pathlib
from src.loader import load

MAT = pathlib.Path("data/YelpChi.mat")

@pytest.mark.skipif(not MAT.exists(), reason="dataset not downloaded")
def test_load_subset(tmp_path):
    db = tmp_path / "t.db"
    load(str(MAT), str(db), n=2000)
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 2000
    assert con.execute("SELECT COUNT(*) FROM edges WHERE relation='rur'").fetchone()[0] > 0
    labels = {r[0] for r in con.execute("SELECT DISTINCT label FROM reviews")}
    assert labels <= {0, 1}
