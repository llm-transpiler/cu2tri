# Changelog

## [1.1.0] - 2025-10-03

### Added
- **GPU Configuration File Support** (`gpu_resource.yml`)
  - Define available GPUs in YAML configuration
  - Support for GPU ID mappings (nvidia-smi, CUDA_VISIBLE_DEVICES, logical)
  - Auto-register GPUs on startup from configuration
  - Per-GPU settings (mode, memory threshold)
  - GPU UUID support for reliable identification

- **GPU ID Mapping System**
  - Handle different GPU ID schemes (nvidia-smi vs CUDA)
  - Automatic mapping in task execution
  - Configurable mappings per GPU

- **GPU Configuration Loader** (`gpu_config_loader.py`)
  - Load and validate GPU configurations from YAML
  - Query GPU mappings at runtime
  - Support for enabled/disabled GPUs

- **Documentation**
  - `GPU_CONFIG_GUIDE.md` - Comprehensive GPU configuration guide
  - Configuration examples for various scenarios
  - Troubleshooting guide for GPU mapping issues

- **Test Scripts**
  - `test_gpu_mapping.py` - Verify GPU mapping configuration

### Changed
- **Severe Error Handling**
  - Reduced pause duration from 5 minutes to **1 minute** (60 seconds)
  - Changed to **automatic recovery** after pause duration
  - Previous: Required manual intervention via `/gpus/clear_error`
  - Now: Auto-resumes scheduling after 60 seconds

- **Server Initialization**
  - Now supports `--gpu-config` parameter for configuration file
  - Auto-registers GPUs from config file on startup
  - Command-line `--gpus` still works and overrides config file

- **Task Execution**
  - Now uses mapped CUDA_VISIBLE_DEVICES from configuration
  - Supports different GPU ID schemes transparently

- **GPU Monitoring**
  - Now uses mapped nvidia-smi IDs from configuration
  - More reliable GPU status queries

### Dependencies
- Added `pyyaml>=6.0` for YAML configuration support

### Configuration
- New default config file: `gpu_resource.yml`
- Updated `config.py`:
  - `error_pause_duration: 60` (was 300)

### Migration Guide

If upgrading from version 1.0.0:

1. **Install new dependency:**
   ```bash
   pip install pyyaml>=6.0
   ```

2. **Create GPU configuration file** (optional but recommended):
   ```bash
   # Use the provided gpu_resource.yml as template
   cp gpu_resource.yml.example gpu_resource.yml
   # Edit to match your system
   ```

3. **Update startup command:**
   ```bash
   # Old way (still works)
   python main.py --gpus 0 1

   # New way (recommended)
   python main.py  # Uses gpu_resource.yml
   ```

4. **Severe error behavior changed:**
   - Now auto-resumes after 60 seconds
   - If you want different duration, edit `config.py`:
     ```python
     error_pause_duration: int = 120  # 2 minutes
     ```

### Backward Compatibility

- All existing command-line parameters still work
- If no `gpu_resource.yml` exists, falls back to manual GPU registration
- API endpoints unchanged
- Task submission format unchanged

---

## [1.0.0] - 2025-10-03

### Initial Release

- GPU task scheduling server
- GPU modes: exclusive/shared
- Task types: functional/performance
- Memory threshold management
- Severe error detection and handling
- REST API for GPU and task management
- Python client library
- Comprehensive documentation
- Test scripts and examples

