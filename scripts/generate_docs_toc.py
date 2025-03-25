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

"""
Even in recursive mode, Sphinx autosummary doesn't create separate pages for
each function / attribute. Let's generate these ourselves. See:
https://stackoverflow.com/a/62613202/101923 and
https://github.com/sphinx-doc/sphinx/issues/7912
"""

import collections
import inspect
import pathlib

import bigframes.bigquery
import bigframes.geopandas
import bigframes.pandas

REPO_ROOT = pathlib.Path(__file__).parent.parent
REFERENCE = REPO_ROOT / "docs" / "reference"
BIGFRAMES_BIGQUERY_DIR = REFERENCE / "bigframes.bigquery"
BIGFRAMES_BIGQUERY_TEMPLATE = """===========================
BigQuery built-in functions
===========================
.. currentmodule:: bigframes.bigquery

.. autosummary::
    :toctree: generated/

"""

BIGFRAMES_GEOPANDAS_DIR = REFERENCE / "bigframes.geopandas"
BIGFRAMES_GEOPANDAS_TEMPLATE = """===============================
BigQuery DataFrames (geopandas)
===============================
.. currentmodule:: bigframes.geopandas

.. autosummary::
    :toctree: generated/

"""

BIGFRAMES_PANDAS_DIR = REFERENCE / "bigframes.pandas"
BIGFRAMES_PANDAS_TEMPLATE = """============================
BigQuery DataFrames (pandas)
============================
.. currentmodule:: bigframes.pandas

.. autosummary::
    :toctree: generated/

"""


excluded_class_attributes = frozenset(
    {
        "__annotations__",
        "__class__",
        "__delattr__",
        "__dict__",
        "__dir__",
        "__doc__",
        "__format__",
        "__getattr__",
        "__getattribute__",
        "__getnewargs__",
        "__hash__",
        "__init__",
        "__init_subclass__",
        "__module__",
        "__new__",
        "__reduce__",
        "__reduce_ex__",
        "__setattr__",
        "__slots__",
        "__subclasshook__",
        "__weakref__",
    }
)


def find_class_attributes(cls, name):
    # Use __dict__ instead of dir() to omit methods only implemented in a superclass.
    # See: https://stackoverflow.com/a/7752095/101923
    for attribute in cls.__dict__:
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


def create_toc_lines(module):
    lines = []
    attributes = collections.deque(
        sorted(module.__all__, key=lambda value: value.casefold())
    )

    while len(attributes) > 0:
        attribute = attributes.pop()
        if attribute in excluded:
            continue

        lines.append(f"    {attribute}\n")

        # Avoid trying to call getattr if this is an attribute on a class, not the
        # module.
        if "." in attribute:
            continue

        value = getattr(module, attribute)
        if inspect.isclass(value):
            for class_attribute in find_class_attributes(value, attribute):
                attributes.appendleft(class_attribute)

    return lines


def generate_module_docs(docspath, module, template):
    lines = create_toc_lines(module)

    with open(docspath / "index.rst", "w", encoding="utf-8") as docs_file:
        docs_file.write(template + "".join(lines))

    for file_path in (docspath / "generated").glob("*.rst"):
        file_path.unlink()


def main():
    generate_module_docs(
        BIGFRAMES_BIGQUERY_DIR, bigframes.bigquery, BIGFRAMES_BIGQUERY_TEMPLATE
    )
    generate_module_docs(
        BIGFRAMES_GEOPANDAS_DIR, bigframes.geopandas, BIGFRAMES_GEOPANDAS_TEMPLATE
    )
    generate_module_docs(
        BIGFRAMES_PANDAS_DIR, bigframes.pandas, BIGFRAMES_PANDAS_TEMPLATE
    )


if __name__ == "__main__":
    main()
