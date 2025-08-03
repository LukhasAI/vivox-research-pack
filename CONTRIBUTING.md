# Contributing to VIVOX

**⚠️ IMPORTANT: VIVOX is a PRIVATE RESEARCH PREVIEW ⚠️**

This software is not open for public contributions. All modifications require explicit written permission from LUKHAS AI.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints

## Authorized Research Partners Only

If you are an authorized research partner and have been granted permission to contribute:

### 1. Contact Requirements
- All contributions must be pre-approved by LUKHAS AI
- Email research@lukhas-ai.com before making any changes
- Include your authorization code in all communications

### 2. Reporting Issues

- Use GitHub Issues to report bugs or suggest features
- Check existing issues before creating a new one
- Provide clear, detailed descriptions
- Include reproduction steps for bugs
- Add relevant labels

### 2. Submitting Code

#### Setup Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR-USERNAME/vivox.git
cd vivox

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify setup
pytest
```

#### Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards

3. Add tests for new functionality

4. Run tests and linting:
   ```bash
   pytest
   black vivox/ tests/
   flake8 vivox/
   mypy vivox/
   ```

5. Commit with descriptive messages:
   ```bash
   git commit -m "feat: add new consciousness state analyzer"
   ```

6. Push and create a Pull Request

### 3. Pull Request Guidelines

- Keep PRs focused on a single feature/fix
- Update documentation as needed
- Ensure all tests pass
- Add tests for new features
- Follow the PR template
- Link related issues

### 4. Coding Standards

#### Python Style
- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use type hints where appropriate
- Document all public functions/classes

#### Naming Conventions
- Classes: `CamelCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

#### Documentation
- Use Google-style docstrings
- Include examples in docstrings
- Update README for significant changes
- Add inline comments for complex logic

### 5. Testing Guidelines

- Write tests for all new features
- Maintain >90% code coverage
- Use pytest for testing
- Include unit and integration tests
- Test edge cases and error conditions

Example test:
```python
import pytest
from vivox import ActionProposal

class TestActionProposal:
    def test_creation(self):
        action = ActionProposal(
            action_type="test",
            content={"data": "value"},
            context={}
        )
        assert action.action_type == "test"
        assert action.content["data"] == "value"
```

### 6. Documentation

- Update docstrings for API changes
- Update README for new features
- Add examples for complex functionality
- Update the whitepaper for theoretical changes

### 7. Component-Specific Guidelines

#### Consciousness (CIL)
- Ensure state transitions are valid
- Maintain coherence calculations
- Test drift detection thoroughly

#### Ethics (MAE)
- Verify ethical principles are respected
- Test precedent matching
- Ensure harmful actions are blocked

#### Memory (ME)
- Test memory folding operations
- Verify emotional encoding
- Check cascade prevention

#### Self-Reflection (SRM)
- Ensure audit trails are complete
- Test report generation
- Verify pattern detection

## Release Process

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Create release PR
4. Tag release after merge
5. Build and publish to PyPI

## Getting Help

- Join our Discord community
- Check the documentation
- Ask questions in GitHub Discussions
- Email: contributors@lukhas-ai.com

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to VIVOX!