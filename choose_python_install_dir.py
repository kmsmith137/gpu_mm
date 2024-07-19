# Invoked from Makefile, to choose python install dir.

import os
import sys
import site

for d in site.getsitepackages() + [ site.getusersitepackages() ]:
    if not os.path.isdir(d):
        print(f"# Skipping python install dir {d}: doesn't exist (or non-directory)", file=sys.stderr)
    elif not os.access(d, os.W_OK):
        print(f"# Skipping python install dir {d}: don't have write access", file=sys.stderr)
    else:
        print(f"# Using python install dir {d}", file=sys.stderr)
        print(d)
        sys.exit(0)

print("No python install directory could be found!", file=sys.stderr)
sys.exit(1)

    
