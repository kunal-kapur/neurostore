#!/bin/bash

# Define the accepted values
accepted_values=("major" "minor" "patch")

variable_to_check=$1

is_accepted=false

for value in "${accepted_values[@]}"; do
    if [ "$variable_to_check" == "$value" ]; then
        is_accepted=true
        break
    fi
done
if [ $is_accepted = false ]; then
    echo "Enter one of 'major', 'minor', or 'patch'"
    exit 1
fi

prev_version=`cat VERSION`

python bump_version.py --type $1

new_version=`cat VERSION`

echo "Bumping ${prev_version} to ${new_version}"
python3 -m build --sdist .
python3 -m build --wheel .

twine upload dist/neurostore-${new_version}.tar.gz dist/neurostore-${new_version}-py3-none-any.whl
git tag $new_version
git add VERSION
git commit -m "Bump version to ${new_version}"
git push --follow-tags