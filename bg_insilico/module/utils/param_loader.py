# -*- coding: utf-8 -*-
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)

import json

# Load JSON parameters
def load_params(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)