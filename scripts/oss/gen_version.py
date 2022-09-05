import argparse
import json
import re

"""
Input file should be like following format:
* only master / stable and losts of 1.x
* stable must point to a commit which exactly one version point to it

{
    "master": "commit-13309e5",
    "stable": "commit-8824f03",
    "1.10": "commit-45492f9",
    "1.11": "commit-8824f03",
    ...
}

Output file will be like:

{
    "en":[
        {
            "name": "master",
            "path": "master",
            "alias": []
        },
        {
            "name": "1.11",
            "path": "1.11",
            "default": true,
            "alias": ["stable"]
        },
        {
            "name": "1.10",
            "path": "1.10",
            "alias": []
        },
    ],
    "zh": ...
}
"""


def main():
    parser = argparse.ArgumentParser(description="Generate version.json for MegEngine Documentation from mapping.json")
    parser.add_argument("-f", "--input", default="mapping.json", type=str, help="Input mapping file")
    parser.add_argument("-v", "--output", default="version.json", type=str, help="Output mapping file")

    args = parser.parse_args()
    run(args.input, args.output)


def get_numeric_ver(v):
    return tuple(map(int, v.split('.')))

def generate_version_for_single_language(mapping):
    assert "master" in mapping
    mapping.pop("master")
    assert "stable" in mapping
    stable_commit = mapping.pop("stable")

    stable_version = None

    output = [{"name": "master", "path": "master", "alias": []}]

    # Must sort version in DESC order
    versions = sorted(list(mapping.items()), key=lambda x: get_numeric_ver(x[0]), reverse=True)

    for k, v in versions:
        d = {"name": k, "path": k, "alias": []}
        assert re.match(r'1\.\d+', k), "version must be in pattern of 1.x"
        if v == stable_commit:
            assert stable_version is None, "There should be only 1 stable commit"
            stable_version = k
            d["alias"].append("stable")
            d["default"] = True
        output.append(d)
    return output

def run(mapping_path, version_path):
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    single_version = generate_version_for_single_language(mapping)
    print("version data:", single_version)

    version_data = {
        "en": single_version,
        "zh": single_version,
    }

    with open(version_path, 'w') as f:
        json.dump(version_data, f, indent=4)

if __name__ == "__main__":
    main()
