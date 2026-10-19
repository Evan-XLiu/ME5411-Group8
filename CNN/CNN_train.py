"""YJN CNN entry point using the shared training and evaluation pipeline."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

if __name__ == "__main__":
    from task_05.cnn import main
    if "--architecture" not in sys.argv:
        sys.argv.extend(["--architecture", "yjn"])
    main()
