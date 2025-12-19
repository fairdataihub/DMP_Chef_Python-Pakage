# 1) Clone the repository
git clone https://github.com/fairdataihub/dmpchef.git
cd dmpchef

# 2) Open the project in VS Code
code .

# 3) Create a new Conda environment (Python 3.10)
# Option A (recommended): named environment
conda create -n dmpchef python=3.10 -y
conda activate dmpchef

# Option B: environment in a specific folder (path-based)
# conda create -p ./venv python=3.10 -y
# conda activate ./venv

# 4) Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5) Install the project (if setup.py exists)
python setup.py install
# OR (more modern / recommended)
# pip install -e .

# 6) Run the web app
uvicorn app:app --reload

