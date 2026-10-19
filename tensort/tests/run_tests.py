"""Run each off-device suite in isolation because suites install import stubs.

Usage: python3 tests/run_tests.py [test_name.py ...]
No cameras or GPU are accessed.
"""

from pathlib import Path
import subprocess
import sys


def main():
    directory = Path(__file__).resolve().parent
    suites = [directory / name for name in sys.argv[1:]] if sys.argv[1:] else sorted(directory.glob("test_*.py"))
    failed = []
    for suite in suites:
        print("Running {}".format(suite.name), flush=True)
        result = subprocess.run([sys.executable, str(suite)], cwd=str(directory.parent),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            failed.append(suite.name)
            print(result.stdout, flush=True)
        else:
            print("  PASS", flush=True)
    print("{} passed; {} failed".format(len(suites) - len(failed), len(failed)))
    if failed:
        print("Failed: " + ", ".join(failed))
    return bool(failed)


if __name__ == "__main__":
    sys.exit(main())
