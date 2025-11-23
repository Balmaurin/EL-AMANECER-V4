#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launcher script to run the full EL-AMANECERV3 integrated system demo.
This script sets up the Python path, imports the unified demo, and executes it.
"""

import sys
import os
import asyncio

# Add the sheily-core source directory to PYTHONPATH so imports work
sys.path.append(os.path.abspath("packages/sheily-core/src"))

# Import the demo function
from demo_unified_system import demo_unified_system

def main():
    print("\n🚀 Iniciando EL-AMANECERV3 - Sistema Completo...\n")
    asyncio.run(demo_unified_system())

if __name__ == "__main__":
    main()
