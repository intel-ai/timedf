import sys
from run_modin_tests import main


if __name__ == "__main__":
    sys.argv[0] = "run_modin_tests.py"
    command_line = []
    if "-task" not in sys.argv:
        command_line.extend(["-task", "benchmark"])
    command_line.extend(sys.argv[1:])
    print("CMD: ", command_line)
    main(command_line)
