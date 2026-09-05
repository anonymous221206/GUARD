# DrugBAN environment installation

Created: 2026-09-04
Base interpreter: /vol/dl-nguyenb5-solar/users/hoangpc/CURA/_home/uv/python/cpython-3.9.25-linux-x86_64-gnu/bin/python3.9
Venv: /vol/dl-nguyenb5-solar/users/hoangpc/GUARD/envs/drugban
Execution: CPU only; every host forward is run with CUDA_VISIBLE_DEVICES empty.

## Package sources and commands

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.1.2
python -m pip install --find-links https://data.dgl.ai/wheels/torch-2.1/repo.html dgl==2.2.1
python -m pip install dgllife==0.3.2 pandas scikit-learn rdkit yacs prettytable
python -m pip install 'numpy<2' 'torchdata==0.7.1' pydantic

Indexes:
- PyPI (default): https://pypi.org/simple
- PyTorch CPU: https://download.pytorch.org/whl/cpu
- DGL Torch-2.1 wheel repository: https://data.dgl.ai/wheels/torch-2.1/repo.html

## Exact resolved packages

annotated-types==0.7.0
certifi==2026.7.22
charset-normalizer==3.5.1
cloudpickle==3.1.2
dgl==2.2.1
dgllife==0.3.2
filelock==3.19.1
fsspec==2025.10.0
future==1.0.0
hyperopt==0.2.7
idna==3.19
Jinja2==3.1.6
joblib==1.5.3
MarkupSafe==3.0.2
mpmath==1.3.0
networkx==3.2.1
numpy==1.26.4
packaging==26.3
pandas==2.3.3
pillow==11.3.0
prettytable==3.16.0
psutil==7.2.2
py4j==0.10.9.9
pydantic==2.13.5
pydantic_core==2.46.5
python-dateutil==2.9.0.post0
pytz==2026.3.post1
PyYAML==6.0.3
rdkit==2025.9.2
requests==2.32.5
scikit-learn==1.6.1
scipy==1.13.1
six==1.17.0
sympy==1.14.0
threadpoolctl==3.6.0
torch==2.1.2+cpu
torchdata==0.7.1
tqdm==4.70.0
typing-inspection==0.4.2
typing_extensions==4.16.0
tzdata==2026.3
urllib3==2.6.3
wcwidth==0.8.3
yacs==0.1.8
