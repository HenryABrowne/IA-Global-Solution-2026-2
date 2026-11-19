"""Copia alguns recortes do conjunto de TESTE para demo/ (exemplos do app)."""
from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "reports" / "split.csv")
test = df[df.split == "test"].sample(frac=1, random_state=3)
out = ROOT / "demo"
out.mkdir(exist_ok=True)
for lab, name, n in [(1, "com_fissura", 3), (0, "sem_fissura", 3)]:
    for i, p in enumerate(test[test.label == lab].path.head(n)):
        shutil.copy(p, out / f"exemplo_{name}_{i + 1}.jpg")
print(sorted(x.name for x in out.glob("exemplo_*")))
