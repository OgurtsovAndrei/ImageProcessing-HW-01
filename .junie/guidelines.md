### Project Guidelines

#### Build and Configuration

- This project uses **Pipenv** for dependency management.
- To set up the environment, run:
  ```bash
  pipenv install
  ```
- To activate the virtual environment, use:
  ```bash
  pipenv shell
  ```

#### Testing Information

- This is a Machine Learning (ML) project.
- No automated tests are currently implemented.

#### Development Workflow and Code Style

- **Decision Confidence & Clarification**:
    - When making any architectural decision, interpreting a requirement, or implementing logic, evaluate your
      confidence on a scale from 0 to 10.
    - If the confidence level is below 8, you must ask the user for clarification before continuing.
    - This may include:
        * clarifying questions about requirements,
        * questions about preferred implementation details,
        * a request for explicit approval of the proposed approach.
        * Do not proceed based on assumptions when confidence is below 8. Stop and align with the user first.
    - If the confidence level is 8 or higher, you may proceed with the implementation without additional clarification.
- **Type Hints**: Always add type hints to all function and method definitions.
- **MyPy Verification**: Verify your changes using MyPy before submitting. You can run it with:
  ```bash
  mypy .
  ```
- **Comments in the code**: We do not add comments to the code.
- **Variable Initialization**: When initializing variables, it is preferred to add class labels (e.g.,
  `variable: ClassLabel = value`).
- **Version Control**: After each successful step in development, perform a Git commit to track progress. Before git
  commit, run linter and mypy.
- **MyPy Configuration**: The project includes a `mypy.ini` file that defines specific strictness rules and exclusions
  for certain directories (e.g., `classification-base/`, `covid-segmantation/tutorials/`). Ensure your code complies
  with these settings.
- **Linting**: Use `flake8` for linting. Run it with:
  ```bash
  flake8 .
  ```
