#!/usr/bin/env python3
import subprocess, sys

subprocess.check_call([
    sys.executable, "run_modelB_deit.py",
    "--config", "configs/VeRi/deit_transreid_stride_b2_supcon_tripletx.yml"
])
