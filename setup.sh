#! /bin/bash
virtualenv --system-site-packages ./venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py install
