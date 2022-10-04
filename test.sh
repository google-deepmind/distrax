# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Runs CI tests on a local machine.
set -xeuo pipefail

# Install deps in a virtual env.
readonly VENV_DIR=/tmp/distrax-env
rm -rf "${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

# Install dependencies.
pip install --upgrade pip setuptools wheel
pip install flake8 pytest-xdist pytest-forked pytype pylint pylint-exit
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-tests.txt

# Lint with flake8.
flake8 `find distrax -name '*.py' | xargs` --count --select=E9,F63,F7,F82,E225,E251 --show-source --statistics

# Lint with pylint.
# Fail on errors, warning, conventions and refactoring messages.
PYLINT_ARGS="-efail -wfail -cfail -rfail"
# Download Google OSS config.
wget -nd -v -t 3 -O .pylintrc https://google.github.io/styleguide/pylintrc
# Append specific config lines.
echo "disable=abstract-method,unnecessary-lambda-assignment,no-value-for-parameter,use-dict-literal" >> .pylintrc
# Lint modules and tests separately.
# Disable `abstract-method` warnings.
pylint --rcfile=.pylintrc `find distrax -name '*.py' | grep -v 'test.py' | xargs` || pylint-exit $PYLINT_ARGS $?
# Disable `protected-access` and `arguments-differ` warnings for tests.
pylint --rcfile=.pylintrc `find distrax -name '*_test.py' | xargs` -d W0212,W0221 || pylint-exit $PYLINT_ARGS $?
# Cleanup.
rm .pylintrc

# Build the package.
python setup.py sdist
pip wheel --verbose --no-deps --no-clean dist/distrax*.tar.gz
pip install distrax*.whl

# Use TFP nightly builds in tests.
pip uninstall tensorflow-probability -y
pip install tfp-nightly

# Check types with pytype.
pytype `find distrax/_src/ -name "*py" | xargs` -k

# Run tests using pytest.
# Change directory to avoid importing the package from repo root.
mkdir _testing && cd _testing

# Main tests.

# Disable JAX optimizations to speed up tests.
export JAX_DISABLE_MOST_OPTIMIZATIONS=True
pytest -n"$(grep -c ^processor /proc/cpuinfo)" --forked `find ../distrax/_src/ -name "*_test.py" | sort` -k 'not _float64_test'

# Isolate tests that set double precision.
pytest -n"$(grep -c ^processor /proc/cpuinfo)" --forked `find ../distrax/_src/ -name "*_test.py" | sort` -k '_float64_test'
unset JAX_DISABLE_MOST_OPTIMIZATIONS

cd ..

set +u
deactivate
echo "All tests passed. Congrats!"
