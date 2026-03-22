from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_quant.cli.quality import main


def _enable_verbose_default() -> None:
    if "--verbose" not in sys.argv:
        sys.argv.append("--verbose")


if __name__ == "__main__":
    print("[script] quality start")
    _enable_verbose_default()
    main()
    print("[script] quality complete")
