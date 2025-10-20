# Contributing to MoodSync AI Service

We welcome contributions to the MoodSync AI Service! This document provides guidelines for contributing to this project.

## 🤝 How to Contribute

### Reporting Issues

1. **Search existing issues** to avoid duplicates
2. **Use the issue template** (when available)
3. **Provide detailed information**:
   - Steps to reproduce the issue
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Relevant logs or error messages

### Suggesting Features

1. **Check existing feature requests** to avoid duplicates
2. **Open a new issue** with the "enhancement" label
3. **Describe the feature** clearly:
   - What problem does it solve?
   - How should it work?
   - Any implementation ideas?

### Code Contributions

#### Getting Started

1. **Fork the repository**
2. **Clone your fork**:

   ```bash
   git clone https://github.com/your-username/moodsync-ai-service-.git
   cd moodsync-ai-service-
   ```

3. **Set up the development environment**:

   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Copy environment template
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Making Changes

1. **Follow the coding style**:
   - Use Python PEP 8 style guidelines
   - Add docstrings to functions and classes
   - Use meaningful variable and function names
   - Keep functions focused and small

2. **Write tests** (when applicable):
   - Add unit tests for new functions
   - Test edge cases and error conditions
   - Ensure tests pass before submitting

3. **Update documentation**:
   - Update README.md if adding new features
   - Add docstrings to new code
   - Update API documentation for new endpoints

#### Submitting Changes

1. **Commit your changes**:

   ```bash
   git add .
   git commit -m "Add feature: brief description of changes"
   ```

2. **Push to your fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request**:
   - Use a clear, descriptive title
   - Reference any related issues
   - Provide a detailed description of changes
   - Include screenshots if applicable

## 📋 Pull Request Guidelines

### Before Submitting

- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] No sensitive data (API keys, passwords) in code
- [ ] Commit messages are clear and descriptive

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] Manual testing completed
- [ ] All tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No sensitive data included
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=app

# Run specific test file
python -m pytest test_specific.py
```

### Test Guidelines

- Write tests for new functionality
- Test both success and failure cases
- Mock external API calls
- Keep tests independent and isolated
- Use descriptive test names

## 🎨 Code Style

### Python Style Guide

- Follow PEP 8 guidelines
- Use 4 spaces for indentation
- Maximum line length: 88 characters
- Use type hints where appropriate
- Order imports: standard library, third-party, local

### Example Code Style

```python
import os
from typing import Dict, List, Optional

import requests
from flask import Flask, jsonify

from .utils import helper_function


class ExampleService:
    """Example service class with proper docstring."""
    
    def __init__(self, api_key: str) -> None:
        """Initialize the service with API key.
        
        Args:
            api_key: The API key for authentication
        """
        self.api_key = api_key
    
    def process_data(self, data: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Process incoming data with proper error handling.
        
        Args:
            data: Dictionary containing input data
            
        Returns:
            Processed data dictionary or None if processing fails
        """
        try:
            # Implementation here
            return processed_data
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return None
```

## 🔒 Security Guidelines

### Sensitive Data

- **Never commit** API keys, passwords, or other secrets
- Use environment variables for configuration
- Add sensitive files to `.gitignore`
- Use placeholder values in examples

### Code Security

- Validate all input data
- Use parameterized queries if working with databases
- Implement proper error handling
- Log security-relevant events

## 📚 Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Use clear, descriptive variable names
- Comment complex logic
- Include type hints

### README Updates

When adding new features, update:

- Feature list
- API documentation
- Setup instructions
- Example usage

## 🌟 Recognition

Contributors will be recognized in several ways:

- Listed in the project's contributors
- Mentioned in release notes for significant contributions
- GitHub contributor statistics

## 🤔 Questions?

If you have questions about contributing:

1. Check existing issues and discussions
2. Open a new issue with the "question" label
3. Reach out to maintainers

## 📄 License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## 🙏 Thank You

Thank you for contributing to MoodSync AI Service! Your efforts help make mental health support more accessible and effective for everyone.

---

**Happy Contributing!** 🚀
