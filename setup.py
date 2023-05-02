# Copyright 2022 Google LLC
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

import io
import itertools
import os
from typing import Dict

import setuptools

# Package metadata.

name = "bigframes"
description = "Scalable DataFrames with BigQuery"

# Should be one of:
# 'Development Status :: 3 - Alpha'
# 'Development Status :: 4 - Beta'
# 'Development Status :: 5 - Production/Stable'
release_status = "Development Status :: 3 - Alpha"
dependencies = [
    "cloudpickle >= 2.2.1",
    "fsspec >=2023.3.0",
    "gcsfs >=2023.3.0",
    "geopandas >=0.12.2",
    "google-auth >2.14.1,<3.0dev",
    "google-cloud-bigquery[bqstorage,pandas] >=3.10.0",
    "google-cloud-functions >=1.10.1",
    "google-cloud-storage >=2.0.0",
    # TODO(swast): Compatibility with latest ibis. "suffixes" argument renamed:
    # https://github.com/ibis-project/ibis/commit/3caf3a12469d017428d5e2bb94143185e8770038
    "ibis-framework[bigquery] >=5.0.0, <6.0.0dev",
    "sqlalchemy >=1.4,<2.0",
    "pandas >=1.5.0",
]
extras = {
    "tests": [
        "pandas-gbq >=0.19.0",
        "scikit-learn >=1.2.2",
    ]
}
extras["all"] = set(itertools.chain.from_iterable(extras.values()))

# Setup boilerplate below this line.

package_root = os.path.abspath(os.path.dirname(__file__))

readme_filename = os.path.join(package_root, "README.rst")
with io.open(readme_filename, encoding="utf-8") as readme_file:
    readme = readme_file.read()

version: Dict[str, str] = {}
with open(os.path.join(package_root, "bigframes/version.py")) as fp:
    exec(fp.read(), version)
version_id = version["__version__"]

# Only include packages under the 'bigframes' namespace. Do not include tests,
# benchmarks, etc.
packages = [
    package
    for package in setuptools.PEP420PackageFinder.find()
    if package.startswith("bigframes")
]

setuptools.setup(
    name=name,
    version=version_id,
    description=description,
    long_description=readme,
    author="Google LLC",
    author_email="googleapis-packages@google.com",
    license="Apache 2.0",
    url="https://github.com/googleapis/python-bigquery",
    classifiers=[
        release_status,
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Internet",
    ],
    install_requires=dependencies,
    extras_require=extras,
    platforms="Posix; MacOS X; Windows",
    packages=packages,
    python_requires=">=3.9",
    include_package_data=True,
    zip_safe=False,
)
