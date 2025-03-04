#!/usr/bin/env python3

import sys
import json as JSON

json = {}

for line in sys.stdin:
    line = line.rstrip()
    parts = line.split('.', 2)
    # print(parts)
    submodule = parts[1].split('/')[1]
    key, val = parts[2].split('=')

    if submodule not in json:
        json[submodule] = {}

    json[submodule][key] = val

    # print(json[submodule])

print(JSON.dumps(json))
