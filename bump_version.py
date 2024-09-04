import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, help='type of change')

update_type = parser.parse_args().type

version_str = ""
with open ("VERSION", 'r') as f:
    version_str = f.read()

major, minor, patch = [int(i) for i in version_str.split(".")]

if update_type == "major":
    major += 1
    minor, patch = 0, 0
elif update_type == "minor":
    minor += 1
    patch = 0
else:
    patch += 1

new_vals = [str(major), str(minor), str(patch)]
with open("VERSION", 'w') as f:
    f.write(".".join(new_vals))






