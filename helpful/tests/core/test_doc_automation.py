"""Tests for documentation automation tools."""

import pytest
import os
import shutil
import tempfile
from datetime import datetime
from src.core.doc_automation import DocAutomation

@pytest.fixture
def temp_project():
    """Create a temporary project directory structure for testing."""
    temp_dir = tempfile.mkdtemp()
    
    # Create project structure
    os.makedirs(os.path.join(temp_dir, 'proteus'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'tests'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'docs'), exist_ok=True)
    
    # Create test Python file with docstrings
    with open(os.path.join(temp_dir, 'proteus', 'test_module.py'), 'w') as f:
        f.write('''"""Test module for documentation automation."""
def test_function(param1, param2):
    """Test function with parameters.
    
    Args:
        param1 (str): First parameter
        param2 (int): Second parameter
        
    Returns:
        bool: Test result
    """
    return True

class TestClass:
    """Test class for documentation automation."""
    
    def test_method(self, param):
        """Test method with parameter.
        
        Args:
            param (float): Test parameter
            
        Returns:
            str: Test result
        """
        return "test"
''')
    
    # Create test file
    with open(os.path.join(temp_dir, 'tests', 'test_test_module.py'), 'w') as f:
        f.write('''"""Tests for test module."""
import pytest
from src.test_module import test_function, TestClass

def test_test_function():
    """Test test_function."""
    assert test_function("test", 1) is True

def test_test_class():
    """Test TestClass."""
    test_obj = TestClass()
    assert test_obj.test_method(1.0) == "test"
''')
    
    # Create initial README
    with open(os.path.join(temp_dir, 'README.md'), 'w') as f:
        f.write('''# Test Project

## TODO
- [ ] Task 1
- [ ] Task 2
''')
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)

def test_initialization(temp_project):
    """Test DocAutomation initialization."""
    doc_automation = DocAutomation(temp_project)
    assert doc_automation.project_root == temp_project
    assert doc_automation.package_name == 'proteus'

def test_update_api_docs(temp_project):
    """Test API documentation update."""
    doc_automation = DocAutomation(temp_project)
    doc_automation.update_api_docs()
    
    api_docs_path = os.path.join(temp_project, 'docs', 'api.md')
    assert os.path.exists(api_docs_path)
    
    with open(api_docs_path, 'r') as f:
        content = f.read()
        assert 'test_function' in content
        assert 'TestClass' in content
        assert 'test_method' in content

def test_update_test_docs(temp_project):
    """Test test documentation update."""
    doc_automation = DocAutomation(temp_project)
    doc_automation.update_test_docs()
    
    test_docs_path = os.path.join(temp_project, 'docs', 'testing.md')
    assert os.path.exists(test_docs_path)
    
    with open(test_docs_path, 'r') as f:
        content = f.read()
        assert 'test_test_function' in content
        assert 'test_test_class' in content

def test_update_changelog(temp_project):
    """Test changelog update."""
    doc_automation = DocAutomation(temp_project)
    doc_automation.update_changelog('Test update')
    
    changelog_path = os.path.join(temp_project, 'CHANGELOG.md')
    assert os.path.exists(changelog_path)
    
    with open(changelog_path, 'r') as f:
        content = f.read()
        assert 'Test update' in content
        assert datetime.now().strftime('%Y-%m-%d') in content

def test_update_readme(temp_project):
    """Test README update."""
    doc_automation = DocAutomation(temp_project)
    doc_automation.update_readme()
    
    readme_path = os.path.join(temp_project, 'README.md')
    assert os.path.exists(readme_path)
    
    with open(readme_path, 'r') as f:
        content = f.read()
        assert '## Features' in content
        assert '## Installation' in content
        assert '## Usage' in content

def test_generate_coverage_report(temp_project):
    """Test coverage report generation."""
    doc_automation = DocAutomation(temp_project)
    doc_automation.generate_coverage_report()
    
    coverage_path = os.path.join(temp_project, 'docs', 'coverage.md')
    assert os.path.exists(coverage_path)

def test_update_all_docs(temp_project):
    """Test updating all documentation."""
    doc_automation = DocAutomation(temp_project)
    doc_automation.update_all_docs()
    
    # Check all documentation files exist
    assert os.path.exists(os.path.join(temp_project, 'docs', 'api.md'))
    assert os.path.exists(os.path.join(temp_project, 'docs', 'testing.md'))
    assert os.path.exists(os.path.join(temp_project, 'CHANGELOG.md'))
    assert os.path.exists(os.path.join(temp_project, 'docs', 'coverage.md'))

def test_error_handling(temp_project):
    """Test error handling."""
    doc_automation = DocAutomation(temp_project)
    
    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        doc_automation.update_api_docs('non_existent.py')
    
    # Test invalid Python file
    invalid_py = os.path.join(temp_project, 'proteus', 'invalid.py')
    with open(invalid_py, 'w') as f:
        f.write('invalid python code')
    
    with pytest.raises(SyntaxError):
        doc_automation.update_api_docs('invalid.py')