# Contriubution guidelines, tailored for LLM agents

## Documentation

If a method or property is implementing the same interface as a third-party
package such as pandas or scikit-learn, place the relevant docstring in the
corresponding `third_party/bigframes_vendored/package_name` directory, not in
the `bigframes` directory. Implementations may be placed in the `bigframes`
directory, though.

## Adding a scalar operator

For an example, see commit c5b7fdae74a22e581f7705bc0cf5390e928f4425.
