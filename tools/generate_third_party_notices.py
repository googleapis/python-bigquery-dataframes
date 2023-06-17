# Copyright 2023 Google LLC
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

import argparse
import importlib.metadata
import json
import os.path
import re
import sys

import piplicenses

THIRD_PARTY_NOTICES_FILE = "THIRD_PARTY_NOTICES"
DEPENDENCY_INFO_SEPARATOR = "*" * 80 + "\n"
PACKAGE_NAME_EXTRACTOR = re.compile("^[a-zA-Z0-9._-]+")


def get_package_dependencies(pkg_name):
    """Get all package dependencies for a given package, both required and optional."""
    packages = set()
    requirements = importlib.metadata.requires(pkg_name)
    if requirements:
        for req in requirements:
            match = PACKAGE_NAME_EXTRACTOR.match(req)
            assert match, f"Could not parse {req} for package name"
            packages.add(match.group(0))
    return packages


# Inspired by third_party/colab/cleanup_filesets.py
def find_dependencies(
    roots: set[str], ignore_missing_metadata=False
) -> dict[str, dict[str, set[str]]]:
    """Return the transitive dependencies of a set of packages.
    Args:
        roots: List of package names, e.g. ["pkg1", "pkg2"]
    Returns:
        A dictionary of dependencies, e.g.
        {
            "pkg3" : {
                "Requires" : set(["pkg4", "pkg5", "pkg6"]),
                "RequiredBy": set(["pkg1"])
            },
            "pkg4" : {
                "Requires" : set([]),
                "RequiredBy": set(["pkg3"])
            },
            ...
        }
    """
    hops = set()
    visited = set()
    deps: dict[str, dict[str, set[str]]] = dict()

    # Initialize the start of the graph walk
    for root in roots:
        # Get the normalized package name
        try:
            pkg = importlib.metadata.metadata(root)
        except importlib.metadata.PackageNotFoundError:
            if not ignore_missing_metadata:
                raise
            continue
        hops.add(pkg["Name"])

    # Start the graph walk
    while True:
        if not hops:
            break
        hop = hops.pop()
        if hop in visited:
            continue
        visited.add(hop)

        for dep in get_package_dependencies(hop):
            # Get the normalized package name
            try:
                req_pkg = importlib.metadata.metadata(dep)
            except importlib.metadata.PackageNotFoundError:
                if not ignore_missing_metadata:
                    raise
                continue
            dep = req_pkg["Name"]

            # Create outgoing edge only for non root packages, for which an
            # entry must have been created in the deps dictionary when we
            # saw the package for the first time during the graph walk
            if hop in deps:
                deps[hop]["Requires"].add(dep)

            if dep in deps:
                # We have already seen this requirement in the graph walk.
                # Just update the incoming dependency and carry on.
                deps[dep]["RequiredBy"].add(hop)
            else:
                # This is the first time we came across this requirement.
                # Create a new entry with the incoming dependency.
                deps[dep] = {"RequiredBy": {hop}, "Requires": set()}

                # Put it in the next hops for further graph traversal
                hops.add(dep)

    return deps


def get_metadata_and_filename(
    package_name: str,
    metadata_name: str,
    metadata_file: str,
    metadata_text: str,
    ignore_missing=True,
) -> tuple[str, str] | None:
    """Get package metadata and corresponsing file name."""

    # Check metadata file
    metadata_filepath_known = metadata_file != piplicenses.LICENSE_UNKNOWN
    if not metadata_filepath_known and not ignore_missing:
        raise ValueError(f"No {metadata_name} file found for {package_name}")

    # Check metadata text
    if metadata_text != piplicenses.LICENSE_UNKNOWN:
        output_filename = metadata_name
        if metadata_filepath_known:
            output_filename = os.path.basename(metadata_file)
        if not output_filename:
            raise ValueError(
                f"Need a file name to write {metadata_name} text for {package_name}."
            )
        return metadata_text, output_filename
    elif not ignore_missing:
        raise ValueError(f"No {metadata_name} text found for {package_name}")

    return None


def fetch_license_and_notice_metadata(packages: list[str]):
    """Fetch metadata including license and notice for given packages.
    Returns a json object.
    """
    parser = piplicenses.create_parser()
    args = parser.parse_args(
        [
            "--format",
            "json",
            "--with-license-file",
            "--with-notice-file",
            "--with-urls",
            "--with-description",
            "--packages",
            *packages,
        ]
    )
    output_str = piplicenses.create_output_string(args)
    metadatas = json.loads(output_str)
    return metadatas


def write_metadata_to_file(
    file, metadata, with_version=False, requires_packages=[], packages_required_by=[]
):
    """Write package metadata to a file object."""
    file.write(DEPENDENCY_INFO_SEPARATOR)

    info_keys = ["Name"]
    if with_version:
        info_keys.append("Version")
    info_keys.extend(["License", "URL"])
    file.writelines([f"{key}: {metadata[key]}\n" for key in info_keys])

    if requires_packages:
        file.write(f"Requires: {', '.join(sorted(requires_packages))}\n")

    if packages_required_by:
        file.write(f"Required By: {', '.join(sorted(packages_required_by))}\n")

    # This is to stop complaints by the trailing-whitespace pre-commit hook
    def write_lines_without_trailing_spaces(text: str, key: str):
        text = "\n".join([line.rstrip() for line in text.split("\n")])
        file.write(f"{key}:\n{text}\n")

    # Try to generate third party license

    # TODO(shobs): There can be packages like 'multipledispatch' which
    # do have a license file
    # https://github.com/mrocklin/multipledispatch/blob/master/LICENSE.txt
    # But it is not available with the package metadata when installed,
    # in other words, it is not found in the installation path such as
    # lib/python3.10/site-packages/multipledispatch-0.6.0.dist-info/.
    # Figure out if we still need to care about including such third
    # party licenses with bigframes.
    # For now ignore such missing licenses via ignore_missing=True

    license_info = get_metadata_and_filename(
        metadata["Name"],
        "LICENSE",
        metadata["LicenseFile"],
        metadata["LicenseText"],
        ignore_missing=True,
    )

    if license_info:
        write_lines_without_trailing_spaces(license_info[0], "License")

    # Try to generate third party notice
    notice_info = get_metadata_and_filename(
        metadata["Name"],
        "NOTICE",
        metadata["NoticeFile"],
        metadata["NoticeText"],
        ignore_missing=True,
    )

    if notice_info:
        write_lines_without_trailing_spaces(notice_info[0], "Notice")

    file.write(DEPENDENCY_INFO_SEPARATOR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate third party notices for bigframes dependencies."
    )
    parser.add_argument(
        "--with-version",
        action="store_true",
        default=False,
        help="Include the version information for each package.",
    )
    parser.add_argument(
        "--with-requires",
        action="store_true",
        default=False,
        help="Include for each package the packages it requires.",
    )
    parser.add_argument(
        "--with-required-by",
        action="store_true",
        default=False,
        help="Include for each package the packages that require it.",
    )
    args = parser.parse_args(sys.argv[1:])

    # Initialize the root package
    roots = {"bigframes"}

    # Find dependencies
    # Let's ignore the packages that are not installed assuming they are
    # just the optional dependencies that bigframes does not require.
    # One example is the dependency path bigframes -> SQLAlchemy -> pg8000,
    # where pg8000 is only an optional dependency for SQLAlchemy which bigframes
    # is not depending on
    # https://github.com/sqlalchemy/sqlalchemy/blob/7bc81947e22dc32368b0c49a41c398cd251d94af/setup.cfg#LL62C21-L62C27
    deps = find_dependencies(roots, ignore_missing_metadata=True)

    # Use third party solution to fetch dependency metadata
    deps_metadata = fetch_license_and_notice_metadata(list(deps))
    deps_metadata = sorted(deps_metadata, key=lambda m: m["Name"])

    # Generate third party metadata for each dependency
    with open(THIRD_PARTY_NOTICES_FILE, "w") as f:
        for metadata in deps_metadata:
            dep = deps[metadata["Name"]]
            write_metadata_to_file(
                f,
                metadata,
                args.with_version,
                dep["Requires"] if args.with_requires else [],
                dep["RequiredBy"] if args.with_required_by else [],
            )
