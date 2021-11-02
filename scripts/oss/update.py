import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser(description="MegEngine Documentation Version Mapping Informaion Updating")
    parser.add_argument("-f", "--filename", default="mapping.json", help="file need to be updated")
    parser.add_argument("-v", "--version", default="master", type=str, help="version of MegEngine")
    parser.add_argument("-c", "--commit_id", default="devtest", type=str, help="commit id")

    args = parser.parse_args()
    update_mapping(args.filename, args.version, args.commit_id)


def update_mapping(filename, version, commit_id):
    assert len(commit_id) == 7
    update_json(filename, version, "commit-" + commit_id)

def update_json(filename, key, value):
    with open(filename, 'r') as f:
        data = json.load(f)
        data[key] = value

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()
