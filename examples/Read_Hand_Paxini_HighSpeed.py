#!/usr/bin/env python3
"""Compatibility entry point for live Paxini tactile streaming.

The repo now uses the smoother local Qt tactile tools only, so this script
forwards to ``read_paxini_tactile.py`` while preserving the familiar filename.

Run:

    python examples/Read_Hand_Paxini_HighSpeed.py
    python examples/Read_Hand_Paxini_HighSpeed.py --port /dev/ttyACM0
    python examples/Read_Hand_Paxini_HighSpeed.py --no-viz --print-rate-hz 5
"""

from __future__ import annotations

from read_paxini_tactile import main


if __name__ == "__main__":
    main()
