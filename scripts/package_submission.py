"""Build a reproducible code/report package without the course dataset."""
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'submission';out.mkdir(exist_ok=True)
with zipfile.ZipFile(out/'Group 08.zip','w',zipfile.ZIP_DEFLATED) as z:
    for folder in ('src','CNN','scripts','tests','report','results/figures','results/experiments'):
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.aux','.log','.out','.fls','.bbl','.blg','.fdb_latexmk','.xdv') and not p.name.endswith('.reference'):
                z.write(p,Path('Group 08')/p.relative_to(ROOT))
    for name in ('README.md','requirements.txt','Makefile','segment.py'):
        z.write(ROOT/name,Path('Group 08')/name)
print(out/'Group 08.zip')
