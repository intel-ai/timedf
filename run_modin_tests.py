# This file is for compatibility with existing CI, because CI is not yet installing timedf
# It will be removed after full CI migration and replaced with direct call to entrypoint along with
# files from build_scripts that need to use that script now
from timedf.scripts.benchmark_run import main


if __name__ == "__main__":
    main()
