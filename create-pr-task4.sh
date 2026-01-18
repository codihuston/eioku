#!/bin/bash

# Create PR for Task 4 - Path Management and Video Discovery
gh pr create \
  --title "Task 4: Path Management and Video Discovery" \
  --body "# Task 4: Path Management and Video Discovery

## 🎯 Task 4.1: Path Configuration Manager - Complete

This PR implements a comprehensive path configuration system for managing video source directories with JSON config file support and command line integration.

### ✅ Implemented Features

#### PathConfigManager Service
- **Full CRUD Operations**: Add, remove, list, get, and update path configurations
- **Duplicate Prevention**: Validates paths before adding to prevent duplicates
- **Database Integration**: Persists configurations using PathConfigRepository
- **Update Support**: Enhanced repository to handle both create and update operations

#### ConfigLoader Service  
- **JSON Configuration**: Loads paths from JSON config files
- **Merge Behavior**: Always adds new config paths to existing database paths (additive only)
- **Filesystem Validation**: Only adds paths that actually exist on the filesystem
- **Graceful Fallback**: Uses built-in defaults if config file is missing or invalid
- **Default Config Creation**: Can generate default configuration files

#### Command Line Integration
- **Config Argument**: \`--config /path/to/config.json\` support
- **Environment Variable**: \`EIOKU_CONFIG_PATH\` override support  
- **System Default**: \`/etc/eioku/config.json\` as default location
- **Built-in Defaults**: \`~/Videos\`, \`/media\`, \`/mnt\` with recursive scanning

#### FastAPI Integration
- **Startup Lifecycle**: Config loading integrated into app startup
- **Dependency Injection**: Proper session management and cleanup
- **Error Handling**: Graceful handling of config loading failures

### 🧪 Test Coverage

**All Tests Passing (10/10):**
- ✅ **ConfigLoader**: 6/6 tests
  - Config file loading and parsing
  - Merge behavior with existing paths  
  - Default config handling
  - Invalid config graceful fallback
  - Default config file creation
  - Existing config file processing
- ✅ **PathConfigManager**: 2/2 tests
  - Database integration with full CRUD operations
  - Mock-based unit tests for service logic
- ✅ **Main App Integration**: 2/2 tests
  - Application startup with config loading
  - Health check endpoints

### 📋 Configuration Behavior

**Startup Process:**
1. Load config from: CLI arg → env var → \`/etc/eioku/config.json\` → defaults
2. Merge config paths with existing database paths (additive only)
3. Skip duplicate paths, add only new paths that exist on filesystem
4. Preserve all existing user-configured paths

**Example Usage:**
\`\`\`bash
# Use custom config
./eioku --config /path/to/my-config.json

# Use environment variable
EIOKU_CONFIG_PATH=/etc/custom.json ./eioku

# Use system default
./eioku  # Uses /etc/eioku/config.json
\`\`\`

**Example Config File:**
\`\`\`json
{
  \"paths\": [
    {\"path\": \"/home/user/Videos\", \"recursive\": true},
    {\"path\": \"/media/external\", \"recursive\": false},
    {\"path\": \"/shared/content\", \"recursive\": true}
  ]
}
\`\`\`

### 🔧 Technical Implementation

#### Clean Architecture
- **Service Layer**: PathConfigManager encapsulates business logic
- **Repository Pattern**: Enhanced PathConfigRepository with update support
- **Domain Models**: PathConfig with business methods
- **Dependency Injection**: Proper IoC with session management

#### Code Quality
- **Type Safety**: Full type annotations with modern Python syntax
- **Error Handling**: Comprehensive validation and graceful fallbacks
- **Testing**: Property-based and integration testing
- **Documentation**: Inline documentation and clear method signatures

### 📊 Requirements Traceability

- ✅ **1.1, 1.2**: Path configuration storage and management
- ✅ **4.7**: Path configuration persistence in database
- ✅ **7.1, 7.2**: Configuration management and deployment support

### 🚀 Ready for Task 4.2

The path configuration system provides a solid foundation for:
- **Video File Discovery**: Scanning configured paths for video files
- **File Validation**: Checking file existence and accessibility
- **Recursive Scanning**: Configurable depth traversal
- **Format Filtering**: Support for MP4, MOV, AVI, MKV formats

This implementation enables both deployment-friendly config file management and runtime path administration through the API, supporting diverse usage scenarios from development to production deployment." \
  --head feature/task-4-path-management \
  --base main
