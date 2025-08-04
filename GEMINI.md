# Contribution guidelines, tailored for LLM agents

## Testing

We use `nox` to instrument our tests.

- To test your changes, run unit tests with `nox`:

  ```bash
  nox -r -s unit
  ```

- To run a single unit test:

  ```bash
  nox -r -s unit-3.13 -- -k <name of test>
  ```

- To run system tests, you can execute::

   # Run all system tests
   $ nox -r -s system

   # Run a single system test
   $ nox -r -s system-3.13 -- -k <name of test>

- The codebase must have better coverage than it had previously after each
  change. You can test coverage via `nox -s unit system cover` (takes a long
  time).

## Code Style

- We use the automatic code formatter `black`. You can run it using
  the nox session `format`. This will eliminate many lint errors. Run via:

  ```bash
  nox -r -s format
  ```

- PEP8 compliance is required, with exceptions defined in the linter configuration.
  If you have ``nox`` installed, you can test that you have not introduced
  any non-compliant code via:

  ```
  nox -r -s lint
  ```

## Documentation

If a method or property is implementing the same interface as a third-party
package such as pandas or scikit-learn, place the relevant docstring in the
corresponding `third_party/bigframes_vendored/package_name` directory, not in
the `bigframes` directory. Implementations may be placed in the `bigframes`
directory, though.

## Adding a scalar operator

For an example, see commit c5b7fdae74a22e581f7705bc0cf5390e928f4425.

## Constraints

- Only add git commits. Do not change git history.
- When following a spec for development, check off the items with `[x]` as they
  are completed.
