# Database Management

## Overview

The database management system in Proteus provides a robust and flexible solution for storing, retrieving, and managing protein structure predictions, model metadata, and analysis results. It supports multiple database backends and provides efficient querying capabilities.

## Key Features

### Storage Management

- Multiple database backend support (SQLite, PostgreSQL, MongoDB)
- Efficient storage of structure predictions
- Model metadata management
- Analysis results storage
- Query optimization

### Data Organization

- Hierarchical data structure
- Flexible schema design
- Data versioning
- Automatic indexing
- Data compression

## Usage

### Basic Database Operations

```python
from src.core.database import DatabaseManager

# Initialize database manager
db_manager = DatabaseManager(
    backend="sqlite",
    database_path="proteus.db"
)

# Store prediction
prediction_id = db_manager.store_prediction(
    sequence="MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
    structure=structure_data,
    metadata={
        "model": "esm2_t33_650M_UR50D",
        "confidence": 0.95
    }
)

# Retrieve prediction
prediction = db_manager.get_prediction(prediction_id)
```

### Advanced Queries

```python
# Query by sequence
predictions = db_manager.query_predictions(
    sequence_pattern="MLS*",
    confidence_threshold=0.8
)

# Query by model
model_predictions = db_manager.query_predictions(
    model_name="esm2_t33_650M_UR50D",
    date_range=("2024-01-01", "2024-03-25")
)
```

## Configuration Options

### Database Configuration

| Option | Description | Default |
|--------|-------------|---------|
| backend | Database backend type | "sqlite" |
| host | Database host | "localhost" |
| port | Database port | Default for backend |
| username | Database username | None |
| password | Database password | None |
| database | Database name | "proteus" |

### Storage Configuration

| Option | Description | Default |
|--------|-------------|---------|
| compression | Enable data compression | True |
| max_connections | Maximum connections | 10 |
| timeout | Connection timeout (s) | 30 |
| retry_attempts | Connection retry attempts | 3 |

## Best Practices

### Data Management

1. Regular database backups
2. Implement data validation
3. Use appropriate indexing
4. Monitor database size
5. Implement data cleanup policies

### Query Optimization

1. Use efficient query patterns
2. Implement caching for frequent queries
3. Batch operations when possible
4. Monitor query performance
5. Use appropriate indexes

## Error Handling

### Database Errors

```python
try:
    db_manager.store_prediction(sequence, structure)
except ConnectionError:
    # Handle connection issues
except ValidationError:
    # Handle data validation issues
except StorageError:
    # Handle storage issues
```

### Query Errors

```python
try:
    results = db_manager.query_predictions(criteria)
except QueryError:
    # Handle query issues
except TimeoutError:
    # Handle timeout issues
```

## Performance Optimization

### Storage Optimization

- Efficient data serialization
- Compression strategies
- Batch operations
- Connection pooling

### Query Optimization

- Index optimization
- Query planning
- Result caching
- Connection reuse

## Integration

### With Model Interface

```python
from src.interfaces import ModelInterface
from src.core.database import DatabaseManager

model = ModelInterface(
    name="esm2_t33_650M_UR50D",
    database_manager=DatabaseManager()
)
```

### With Distributed Systems

```python
from src.core.database import DistributedDatabaseManager

dist_db = DistributedDatabaseManager(
    sharding_strategy="hash",
    replicas=3
)
```

## Monitoring and Metrics

### Database Metrics

- Storage usage
- Query performance
- Connection pool status
- Cache hit rates

### System Metrics

- Memory usage
- I/O operations
- Network latency
- CPU utilization

## Schema Design

### Core Tables

1. Predictions
   - prediction_id (PRIMARY KEY)
   - sequence
   - structure_data
   - metadata
   - created_at

2. Models
   - model_id (PRIMARY KEY)
   - name
   - version
   - configuration

3. Analysis
   - analysis_id (PRIMARY KEY)
   - prediction_id (FOREIGN KEY)
   - results
   - parameters

## Data Migration

### Migration Tools

```python
from src.core.database import DatabaseMigrator

migrator = DatabaseMigrator()
migrator.upgrade_schema()
```

### Backup and Restore

```python
# Backup database
db_manager.backup(path="backup.db")

# Restore database
db_manager.restore(path="backup.db")
```

## API Reference

### Database Manager API

- `store_prediction(sequence, structure, metadata)`
- `get_prediction(prediction_id)`
- `query_predictions(criteria)`
- `update_prediction(prediction_id, data)`
- `delete_prediction(prediction_id)`

### Migration API

- `upgrade_schema()`
- `downgrade_schema(version)`
- `backup(path)`
- `restore(path)`
- `verify_integrity()`
