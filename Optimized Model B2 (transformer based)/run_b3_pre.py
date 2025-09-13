#!/usr/bin/env python3
import subprocess, sys

subprocess.check_call([
    sys.executable, "run_modelB_deit.py",
    "--config", "configs/VeRi/deit_transreid_stride_b3_ssl_pretrain.yml"
])
