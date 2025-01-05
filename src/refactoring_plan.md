# Code Refactoring Plan

## 1. Create Common Measurement Module
**File:** `src/quantum/measurements.py`

### Functions to Move:
- `measure_phase_distance` from QuantumAgent
- `measure_self_awareness` from SelfAwareAgent  
- `measure_collective_awareness` from repository_consciousness
- `_measure_energy_stability` from SelfAwareAgent

### Tasks:
- Add proper type hints
- Add comprehensive docstrings
- Create unit tests
- Ensure consistent return types
- Add error handling

## 2. Update Base Classes

### Modify `src/agents/base.py`:
- Remove duplicated measurement methods
- Import from new measurements module
- Update docstrings and references
- Maintain backwards compatibility

### Update `src/agents/self_aware.py`:
- Remove duplicated methods
- Update to use new measurement module
- Ensure all references are updated
- Verify functionality remains intact

## 3. Create Visualization Module
**File:** `src/visualization/ascii.py`

### Functions to Move:
- `ascii_visualize_agent`
- `ascii_visualize_network`
- `ascii_visualize_awareness_history`

### Tasks:
- Add type hints
- Create standardized interface
- Add documentation
- Ensure consistent styling
- Add customization options

## 4. Update Repository Consciousness
**File:** `src/agents/repository_consciousness.py`

### Tasks:
- Remove duplicated visualization code
- Import from new visualization module
- Update measurement calls
- Verify functionality
- Update documentation

## 5. Testing and Validation

### Unit Tests:
- Create tests for measurements module
- Create tests for visualization module
- Update existing tests
- Add integration tests

### Validation:
- Verify all existing functionality
- Check performance impacts
- Test backwards compatibility
- Review error handling

## Implementation Order:
1. Create new modules
2. Move functions
3. Update existing files
4. Add tests
5. Validate changes
6. Update documentation

## Notes:
- Keep existing functionality intact
- Maintain type safety
- Follow consistent naming conventions
- Add proper error messages
- Document all changes 