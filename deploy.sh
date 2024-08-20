python3 -m build --sdist .
python3 -m build --wheel .
twine upload dist/neurostore-0.1.0.tar.gz dist/neurostore-0.1.0-py3-none-any.whl 