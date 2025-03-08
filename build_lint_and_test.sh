git submodule update --init
git config submodule.external/slaythetext.ignore untracked

# Make sure we are in a venv
if [ -z "$VIRTUAL_ENV" ]; then
    echo "No virtual environment found. Please activate a venv and try again."
    exit 1
fi

# Install dev requirements and package
pip install wheel
pip install -r dev-requirements.txt -e .

echo "Running black..."
black rl_the_spire

echo "Running isort..."
isort --profile black rl_the_spire

echo "Running flake8..."
flake8 rl_the_spire

echo "Running mypy..."
mypy --strict rl_the_spire

echo "Running pytest..."
pytest -s ./rl_the_spire/tests