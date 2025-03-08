# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Even in recursive mode, Sphinx autosummary doesn't create separate pages for
# each function / attribute. Let's generate these ourselves. See:
# https://stackoverflow.com/a/62613202/101923 and
# https://github.com/sphinx-doc/sphinx/issues/7912

import inspect
import pathlib
import shutil

import bigframes.pandas

REPO_ROOT = pathlib.Path(__file__).parent.parent
REFERENCE = REPO_ROOT / "docs" / "reference"
BIGFRAMES_PANDAS = REFERENCE / "bigframes.pandas"

template = """============================
BigQuery DataFrames (pandas)
============================
.. currentmodule:: bigframes.pandas

.. autosummary::
    :toctree: generated/

"""

excluded_class_attributes = frozenset(
    {
        "__annotations__",
        "__init__",
        "__init_subclass__",
        "__new__",
        "__dict__",
        "__dir__",
        "__doc__",
        "__class__",
        "__subclasshook__",
        "__weakref__",
        "__hash__",
        "__module__",
        "__slots__",
    }
)


def find_class_attributes(cls, name):
    for attribute in dir(cls):
        if (attribute in excluded_class_attributes) or (
            attribute.startswith("_") and not attribute.startswith("__")
        ):
            continue

        yield f"{name}.{attribute}"


# Even with excluding the .nox directory in the docs/conf.py file, Sphinx still
# complains when we alias pandas objects directly.
excluded = frozenset(
    {
        "ArrowDtype",
        "BooleanDtype",
        "Float64Dtype",
        "Int64Dtype",
        "StringDtype",
        "NA",
    }
)

lines = []
attributes = list(sorted(bigframes.pandas.__all__, key=lambda value: value.casefold()))

while len(attributes) > 0:
    attribute = attributes.pop()
    if attribute in excluded:
        continue

    lines.append(f"    {attribute}\n")

    # Avoid trying to call getattr if this is an attribute on a class, not the
    # module.
    if "." in attribute:
        continue

    value = getattr(bigframes.pandas, attribute)
    if inspect.isclass(value):
        for class_attribute in find_class_attributes(value, attribute):
            attributes.insert(0, class_attribute)


with open(BIGFRAMES_PANDAS / "index.rst", "w", encoding="utf-8") as docs_file:
    docs_file.write(template + "".join(lines))

shutil.rmtree(BIGFRAMES_PANDAS / "generated")

# TODO: update toc yaml for cloud.google.com docs.
