# GitHub Actions Workflows

This document explains the GitHub Actions workflows implemented in the SPLADE Model Trainer project.

## Dependency Update Testing

We have implemented automated testing for dependency updates to ensure that our codebase remains compatible with updated libraries.

### Workflows

#### 1. Dependency Update Test (`dependency-test.yml`)

This workflow runs whenever the `requirements.txt` file is changed, either via direct commits to the main branch or through pull requests.

- **Trigger**: Changes to `requirements.txt` file
- **Python Versions**: Tests against Python 3.8, 3.9, and 3.10
- **Test Actions**:
  - Basic import checks
  - Code quality verification
  - Running the test suite
  - **Mini training test** to verify model training works
  - Package import validation

#### 2. Dependency Bot Test (`dependency-bot-test.yml`)

This workflow is specifically designed to test pull requests from dependency bots (Dependabot and Renovate).

- **Trigger**: PRs from dependency bots that modify `requirements.txt`
- **Python Versions**: Tests against Python 3.8, 3.9, and 3.10
- **Test Actions**:
  - Full test suite execution
  - **Mini training test** to verify core training functionality
  - Import error detection
  - Basic functionality testing
  - Automated PR commenting with test results

### Dependabot Configuration

Dependabot is configured to:

- Check for updates weekly
- Group related dependencies (Torch, Transformers, etc.)
- Skip major version updates that might introduce breaking changes
- Automatically assign reviewers and labels

## Adding New Tests

When adding new features to the codebase, consider including corresponding tests that can be run by these workflows. This helps ensure that dependency updates won't break your newly added functionality.

## Manual Workflow Execution

If needed, you can manually trigger these workflows from the GitHub Actions tab in the repository.

## Future Enhancements

Planned improvements for our CI/CD pipeline:

1. Add code coverage reporting
2. Implement performance benchmarking
3. Add automatic documentation generation
4. Implement model evaluation checks to test model quality

## Troubleshooting

If you encounter issues with the GitHub Actions workflows:

1. Check the workflow logs in the GitHub Actions tab
2. Verify that your test files are discoverable by the unittest framework
3. Ensure that all dependencies are correctly specified in the requirements.txt file
