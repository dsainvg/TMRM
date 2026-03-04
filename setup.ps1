python -m venv .\.venv
source .\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py > output.log 2>&1
