# Implementation Plan

## Overview

Integrate comprehensive metadata extraction functionality from `comfyui_image_metadata_extension` into `telegram_sender.py` to replace the current basic metadata building system with a robust, Civitai-compatible metadata extraction system.

The goal is to enhance the Telegram sender's metadata capabilities by incorporating the advanced metadata extraction, LoRA handling, embedding support, and hash calculation features from the metadata extension, while maintaining all existing Telegram functionality.

## Types

### New Metadata Classes and Enums

**MetaField Enum**: Defines metadata field types for comprehensive extraction
```python
class MetaField(IntEnum):
    MODEL_NAME = 0
    MODEL_HASH = 1
    VAE_NAME = 2
    VAE_HASH = 3
    POSITIVE_PROMPT = 10
    NEGATIVE_PROMPT = 11
    CLIP_SKIP = 12
    SEED = 20
    STEPS = 21
    CFG = 22
    SAMPLER_NAME = 23
    SCHEDULER = 24
    DENOISE = 26
    IMAGE_WIDTH = 30
    IMAGE_HEIGHT = 31
    EMBEDDING_NAME = 40
    EMBEDDING_HASH = 41
    LORA_MODEL_NAME = 50
    LORA_MODEL_HASH = 51
    LORA_STRENGTH_MODEL = 52
    LORA_STRENGTH_CLIP = 53
    UPSCALE_MODEL_NAME = 80
    UPSCALE_MODEL_HASH = 81
    UPSCALE_BY = 83
```

**MetadataScope Enum**: Controls metadata inclusion levels
```python
class MetadataScope(str, Enum):
    FULL = "full"
    DEFAULT = "default"
    PARAMETERS_ONLY = "parameters_only"
    WORKFLOW_ONLY = "workflow_only"
    NONE = "none"
```

### Enhanced Data Structures

**Capture Field Mapping**: Dictionary mapping ComfyUI node types to metadata field extractions
**PNG Info Dictionary**: Comprehensive metadata dictionary with all generation parameters
**Hash Cache System**: Thread-safe caching for model hashes and file modification times

## Files

### New Files to Create

1. **`metadata_capture.py`** - Metadata capture and processing
   - Capture class with comprehensive metadata extraction
   - Node tracing and filtering logic
   - LoRA string generation and hash calculation
   - Parameter string generation for A1111 compatibility

2. **`metadata_captures.py`** - Node-specific field mappings
   - CAPTURE_FIELD_LIST dictionary
   - Node type to metadata field mappings
   - Validation functions for metadata extraction

3. **`metadata_trace.py`** - Node tracing functionality
   - Trace class for BFS traversal of workflow nodes
   - Node filtering and distance calculation
   - Sampler node identification

### Existing Files to Modify

1. **`telegram_sender.py`** - Integrate metadata extraction
   - Replace `_build_metadata_text()` with comprehensive metadata system
   - Replace `_extract_parameters_from_workflow()` with `Capture.gen_pnginfo_dict()`
   - Add `_extract_loras_from_workflow()` using LoRA extraction system
   - Integrate hash calculation and embedding handling
   - Enhance metadata scope options (full, parameters_only, workflow_only, etc.)

2. **`metadata_utils.py`** - Enhance existing utilities
   - Add MetaField enum
   - Add MetadataScope enum
   - Add embedding extraction functions
   - Add validation functions

### Files to Reference (Not Modify)

1. **`comfyui_image_metadata_extension/modules/`** - Source of metadata extraction logic
   - `capture.py` - Core Capture class
   - `trace.py` - Node tracing functionality  
   - `defs/captures.py` - Field mappings
   - `defs/meta.py` - MetaField enum
   - `defs/formatters.py` - Hash calculation functions

## Functions

### New Functions

1. **`Capture.gen_pnginfo_dict()`** - Main metadata extraction from workflow
2. **`Trace.trace()`** - Node tracing for workflow analysis
3. **`Trace.find_sampler_node_id()`** - Identify sampler nodes
4. **`Capture.get_lora_strings_and_hashes()`** - Process LoRA metadata
5. **`Capture.gen_parameters_str()`** - Generate A1111-style parameter string
6. **`Capture.get_hashes_for_civitai()`** - Generate Civitai-compatible hashes
7. **`Capture.get_sampler_for_civitai()`** - Format sampler names for Civitai
8. **`Capture._collect_all_metadata()`** - Fallback metadata collection

### Modified Functions

1. **`_build_metadata_text()`** - Replace with comprehensive metadata building using `gen_parameters_str()`
2. **`_extract_parameters_from_workflow()`** - Replace with `gen_pnginfo_dict()` from Capture class
3. **`_extract_loras_from_workflow()`** - Enhance with comprehensive LoRA extraction and hash calculation
4. **`send_to_telegram()`** - Add metadata_scope parameter and integrate comprehensive metadata extraction

### Removed Functions

1. Basic parameter extraction functions that are replaced by comprehensive system

## Classes

### New Classes

1. **`Capture`** - Comprehensive metadata capture and processing class
2. **`Trace`** - Node tracing and filtering for metadata extraction
3. **`MetaField`** - Enum defining metadata field types
4. **`MetadataScope`** - Enum controlling metadata inclusion levels

### Modified Classes

1. **`TelegramSender`** - Integrate comprehensive metadata extraction
   - Add metadata_scope parameter to INPUT_TYPES
   - Replace basic metadata building with comprehensive system
   - Enhance LoRA handling with hash calculation
   - Add embedding support
   - Improve parameter extraction accuracy

### Removed Classes

None - existing classes are enhanced rather than removed

## Dependencies

### New Dependencies

1. **`collections`** - defaultdict for metadata collection
2. **`threading`** - Thread-safe hash cache operations
3. **`functools`** - LRU cache for file modification times
4. **`enum`** - Enum support for MetaField and MetadataScope
5. **`re`** - Regular expressions for embedding and LoRA parsing
6. **`json`** - JSON serialization for workflow metadata

### Existing Dependencies

All existing dependencies in `telegram_sender.py` are retained:
- `os`, `re`, `requests`, `threading`, `time`, `json`, `folder_paths`
- `PIL`, `numpy`, `torch`

## Key Integration Points

### Metadata Scope Integration
- Add `metadata_scope` parameter to `TelegramSender.INPUT_TYPES()`
- Implement scope-based metadata filtering in `_build_metadata_text()`
- Support full, parameters_only, workflow_only, and none scopes

### Hash Calculation Integration
- Integrate thread-safe hash caching from `metadata_utils.py`
- Add model, VAE, LoRA, and embedding hash calculation
- Implement file modification tracking for cache invalidation

### LoRA and Embedding Support
- Extract LoRA names and weights from workflow nodes
- Generate Civitai-compatible LoRA strings (`<lora:name:weight>`)
- Extract embedding names and calculate hashes
- Support embedding detection in prompt text

### Node Tracing Integration
- Implement BFS traversal for workflow analysis
- Identify sampler nodes for parameter extraction
- Filter inputs based on node distance and relevance

## Implementation Order

1. **Create metadata core modules**
   - Create `metadata_capture.py` with Capture class (adapted from extension)
   - Create `metadata_trace.py` with Trace class (adapted from extension)
   - Create `metadata_captures.py` with field mappings (adapted from extension)

2. **Enhance metadata utilities**
   - Add MetaField enum to `metadata_utils.py`
   - Add MetadataScope enum to `metadata_utils.py`
   - Add embedding extraction functions
   - Add validation functions

3. **Integrate comprehensive metadata extraction**
   - Replace `_build_metadata_text()` with `Capture.gen_parameters_str()`
   - Replace `_extract_parameters_from_workflow()` with `Capture.gen_pnginfo_dict()`
   - Add metadata scope parameter handling
   - Integrate node tracing for workflow analysis

4. **Enhance LoRA and embedding support**
   - Implement LoRA string generation using `Capture.get_lora_strings_and_hashes()`
   - Add embedding extraction and hashing
   - Update LoRA routing with hash-based matching

5. **Add comprehensive testing**
   - Create unit tests for all new functions
   - Create integration tests for Telegram functionality
   - Validate Civitai compatibility

6. **Performance optimization and final validation**
   - Optimize metadata extraction speed
   - Validate backward compatibility
   - Test with real-world workflows

## Testing Strategy

### Unit Tests
- Test hash calculation for different model types
- Test embedding extraction from prompt text
- Test LoRA string generation and hash calculation
- Test parameter string generation for various workflows

### Integration Tests
- Test metadata scope options (full, parameters_only, workflow_only, none)
- Test enhanced LoRA routing with hash-based matching
- Test embedding support in metadata
- Test compatibility with existing Telegram functionality

### Workflow Compatibility Tests
- Test with various ComfyUI node types
- Test with different sampler configurations
- Test with LoRA and embedding usage
- Test with upscale and VAE configurations

## Validation Strategies

1. **Metadata accuracy validation** - Compare extracted metadata with ground truth
2. **Civitai compatibility validation** - Ensure metadata format matches Civitai expectations
3. **Performance validation** - Ensure metadata extraction doesn't significantly impact send speed
4. **Backward compatibility validation** - Ensure existing Telegram functionality remains intact

## Migration Considerations

### Backward Compatibility
- Maintain existing `TelegramSender` API
- Preserve all existing functionality
- Add new features as optional enhancements

### Performance Impact
- Implement efficient caching for hash calculations
- Use lazy loading for metadata extraction
- Optimize node tracing for large workflows

### Error Handling
- Graceful fallback for missing metadata
- Robust error handling for malformed workflows
- Comprehensive logging for debugging

This implementation plan provides a comprehensive roadmap for integrating the advanced metadata extraction capabilities from the comfyui_image_metadata_extension into the Telegram sender, ensuring Civitai compatibility while maintaining backward compatibility and performance.
