"""Unit tests for logging configuration."""

import unittest
import logging
import json
import os
import tempfile
from pathlib import Path
from datetime import datetime

from multi_model_analysis.utils.logging_config import (
    StructuredFormatter,
    setup_logging,
    get_logger,
    log_with_context
)

class TestLoggingConfig(unittest.TestCase):
    """Test cases for logging configuration."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        # Remove all files in temp directory
        for file in self.log_dir.glob('*'):
            file.unlink()
        os.rmdir(self.temp_dir)
    
    def test_structured_formatter(self):
        """Test structured formatter output."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        # Add extra fields
        record.extra = {'test_field': 'test_value'}
        
        # Format the record
        formatted = formatter.format(record)
        
        # Parse the JSON output
        log_data = json.loads(formatted)
        
        # Check required fields
        self.assertIn('timestamp', log_data)
        self.assertEqual(log_data['level'], 'INFO')
        self.assertEqual(log_data['message'], 'Test message')
        self.assertEqual(log_data['module'], 'test')
        self.assertEqual(log_data['line'], 1)
        self.assertEqual(log_data['test_field'], 'test_value')
    
    def test_setup_logging(self):
        """Test logging setup."""
        # Test with default configuration
        setup_logging(log_dir=str(self.log_dir))
        
        # Check if log files are created
        self.assertTrue((self.log_dir / 'multi_model_analysis.log').exists())
        self.assertTrue((self.log_dir / 'error.log').exists())
        
        # Test with custom configuration
        setup_logging(
            log_dir=str(self.log_dir),
            log_level='DEBUG',
            max_bytes=1024,  # 1KB
            backup_count=2,
            console_output=False
        )
        
        # Write enough logs to trigger rotation
        logger = get_logger('test')
        for _ in range(10):
            logger.debug('Test message')
        
        # Check if backup files are created
        log_files = list(self.log_dir.glob('multi_model_analysis.log*'))
        self.assertGreater(len(log_files), 1)
    
    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger('test')
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'test')
    
    def test_log_with_context(self):
        """Test logging with context."""
        setup_logging(log_dir=str(self.log_dir))
        logger = get_logger('test')
        
        # Test logging with context
        log_with_context(
            logger,
            logging.INFO,
            'Test message',
            {'context_field': 'context_value'}
        )
        
        # Read the log file
        with open(self.log_dir / 'multi_model_analysis.log', 'r') as f:
            log_data = json.loads(f.readline())
        
        # Check context field
        self.assertEqual(log_data['context_field'], 'context_value')
    
    def test_error_logging(self):
        """Test error logging."""
        setup_logging(log_dir=str(self.log_dir))
        logger = get_logger('test')
        
        # Log an error
        try:
            raise ValueError('Test error')
        except ValueError as e:
            logger.error('Error occurred', exc_info=True)
        
        # Check error log file
        with open(self.log_dir / 'error.log', 'r') as f:
            log_data = json.loads(f.readline())
        
        self.assertEqual(log_data['level'], 'ERROR')
        self.assertIn('exception', log_data)
        self.assertIn('ValueError: Test error', log_data['exception'])

if __name__ == '__main__':
    unittest.main() 