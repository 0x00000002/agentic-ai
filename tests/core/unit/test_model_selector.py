"""
Unit tests for the ModelSelector class.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from typing import Dict, Any
from enum import Enum

# Classes and Enums to test/mock
from src.core.model_selector import ModelSelector, UseCase
# Do not import the real ModelEnum here to avoid collection errors
# from src.config.dynamic_models import Model as ModelEnum 
from src.config.unified_config import UnifiedConfig
from src.exceptions import AISetupError

# Define a Mock Enum for use ONLY in this test file
class MockModelEnum(Enum):
    FAST_LOW_ID = "fast-low-id"
    STANDARD_MEDIUM_ID = "standard-medium-id"
    SLOW_HIGH_ID = "slow-high-id"
    LOCAL_MODEL_ID = "local-model-id"
    # Add any other mock enum members needed by tests

# Dummy data for mocking config
DUMMY_MODELS_CONFIG: Dict[str, Dict[str, Any]] = {
    "model-fast-low": {
        "model_id": "fast-low-id",
        "provider": "prov-A",
        "quality": "LOW",
        "speed": "FAST",
        "privacy": "EXTERNAL",
        "cost": {"input_tokens": 0.1, "output_tokens": 0.2, "minimum_cost": 0.01}
    },
    "model-std-med": {
        "model_id": "standard-medium-id",
        "provider": "prov-B",
        "quality": "MEDIUM",
        "speed": "STANDARD",
        "privacy": "EXTERNAL",
        "cost": {"input_tokens": 0.5, "output_tokens": 1.0, "minimum_cost": 0.05}
    },
    "model-slow-high": {
        "model_id": "slow-high-id",
        "provider": "prov-C",
        "quality": "HIGH",
        "speed": "SLOW",
        "privacy": "EXTERNAL",
        "cost": {"input_tokens": 1.0, "output_tokens": 2.0, "minimum_cost": 0.1}
    },
    "model-local": {
        "model_id": "local-model-id",
        "provider": "prov-local",
        "quality": "MEDIUM",
        "speed": "FAST",
        "privacy": "LOCAL",
        "cost": {"input_tokens": 0, "output_tokens": 0} # Assume local is free
    },
}

DUMMY_USE_CASE_CONFIG: Dict[str, Dict[str, Any]] = {
    "chat": {
        "quality": "MEDIUM",
        "speed": "STANDARD",
        "privacy": "EXTERNAL" 
    },
    "coding": {
        "quality": "HIGH",
        "speed": "SLOW",
    },
    "summarization": {
        "quality": "LOW",
        "speed": "FAST",
    },
    "image_generation": { # Example for a use case not covered by default models
        "quality": "HIGH",
        "speed": "SLOW",
    },
    "data_analysis": {
        "quality": "MEDIUM",
        "speed": "FAST",
        "privacy": "LOCAL"
    }
}


@pytest.fixture
def mock_unified_config():
    """Provides a mocked UnifiedConfig instance for testing ModelSelector."""
    with patch('src.core.model_selector.UnifiedConfig.get_instance') as mock_get_instance:
        mock_instance = MagicMock(spec=UnifiedConfig)
        mock_instance.get_all_models.return_value = DUMMY_MODELS_CONFIG
        mock_instance.get_use_case_config.side_effect = lambda uc: DUMMY_USE_CASE_CONFIG.get(uc, {}) # Return {} if use case unknown
        # Mock other methods as needed for specific tests
        mock_get_instance.return_value = mock_instance
        yield mock_instance


# --- Test Suite ---

class TestModelSelector:
    """Test suite for ModelSelector functionality."""

    def test_initialization(self, mock_unified_config):
        """Test that ModelSelector initializes correctly and builds the mapping."""
        selector = ModelSelector()
        assert selector._config is mock_unified_config
        mock_unified_config.get_all_models.assert_called_once()
        # Add checks for the _model_id_to_enum mapping if possible/needed
        # For now, just ensure it doesn't crash
        assert hasattr(selector, '_model_id_to_enum')
        # Example check based on dummy data (adjust if needed)
        # Assuming ModelEnum has values corresponding to DUMMY_MODELS_CONFIG keys
        # Example: assert selector._model_id_to_enum["standard-medium-id"] == ModelEnum.STANDARD_MEDIUM_ID 

    def test_get_system_prompt_known_use_case(self, mock_unified_config):
        """Test getting a system prompt for a known use case."""
        selector = ModelSelector()
        prompt = selector.get_system_prompt(UseCase.CODING)
        assert isinstance(prompt, str)
        assert "expert programmer" in prompt.lower()

    def test_get_system_prompt_unknown_use_case(self, mock_unified_config):
         """Test getting a system prompt returns the default for an unknown use case."""
         selector = ModelSelector()

         # Temporarily add an unknown use case to the enum for testing
         class PatchedUseCase(Enum):
             TRANSLATION = UseCase.TRANSLATION.value
             SUMMARIZATION = UseCase.SUMMARIZATION.value
             CODING = UseCase.CODING.value
             SOLIDITY_CODING = UseCase.SOLIDITY_CODING.value
             CHAT = UseCase.CHAT.value
             CONTENT_GENERATION = UseCase.CONTENT_GENERATION.value
             DATA_ANALYSIS = UseCase.DATA_ANALYSIS.value
             WEB_ANALYSIS = UseCase.WEB_ANALYSIS.value
             IMAGE_GENERATION = UseCase.IMAGE_GENERATION.value
             HYPOTHETICAL_UNKNOWN = 999 # Add a value not in the prompt dict

         # Patch the UseCase enum used within the selector
         with patch('src.core.model_selector.UseCase', PatchedUseCase):
             # Now call with the unknown enum member
             prompt = selector.get_system_prompt(PatchedUseCase.HYPOTHETICAL_UNKNOWN)
             assert prompt == "You are a helpful assistant."
              
    # --- Placeholder Tests for select_model ---
    
    @pytest.mark.parametrize(
        "use_case, expected_model_enum",
        [
            (UseCase.CHAT, MockModelEnum.STANDARD_MEDIUM_ID), 
            (UseCase.CODING, MockModelEnum.SLOW_HIGH_ID),     
            (UseCase.SUMMARIZATION, MockModelEnum.FAST_LOW_ID), 
            (UseCase.DATA_ANALYSIS, MockModelEnum.LOCAL_MODEL_ID), 
        ]
    )
    def test_select_model_basic_use_cases(self, mock_unified_config, use_case, expected_model_enum):
         """Test selecting model based on default use case parameters."""
         selector = ModelSelector()
         
         # Use MockModelEnum here as well
         selector._model_id_to_enum = {
             "fast-low-id": MockModelEnum.FAST_LOW_ID,
             "standard-medium-id": MockModelEnum.STANDARD_MEDIUM_ID,
             "slow-high-id": MockModelEnum.SLOW_HIGH_ID,
             "local-model-id": MockModelEnum.LOCAL_MODEL_ID,
         }
         
         # Patch the actual ModelEnum lookup within select_model to use our MockEnum for the return value comparison
         # This assumes select_model ends by looking up the chosen model_id in _model_id_to_enum
         # or directly converting model_id string to ModelEnum(model_id)
         with patch('src.core.model_selector.ModelEnum', MockModelEnum):
             selected_model = selector.select_model(use_case=use_case)
             assert selected_model == expected_model_enum

    def test_select_model_with_overrides(self, mock_unified_config):
        """Test select_model when explicit parameters override use case defaults."""
        selector = ModelSelector()
        # Use MockModelEnum here
        selector._model_id_to_enum = { 
             "fast-low-id": MockModelEnum.FAST_LOW_ID,
             "standard-medium-id": MockModelEnum.STANDARD_MEDIUM_ID,
             "slow-high-id": MockModelEnum.SLOW_HIGH_ID,
             "local-model-id": MockModelEnum.LOCAL_MODEL_ID, # Include local if needed
        }
        
        with patch('src.core.model_selector.ModelEnum', MockModelEnum):
            # Chat usually defaults to MEDIUM/STANDARD, override to HIGH/SLOW
            selected_model = selector.select_model(
                use_case=UseCase.CHAT, 
                quality="HIGH", 
                speed="SLOW"
            )
            assert selected_model == MockModelEnum.SLOW_HIGH_ID # Compare with MockModelEnum

    def test_select_model_with_cost_constraint(self, mock_unified_config):
        """Test select_model applies cost constraints correctly."""
        selector = ModelSelector()
        # Use MockModelEnum here
        selector._model_id_to_enum = {
            "fast-low-id": MockModelEnum.FAST_LOW_ID,
            "standard-medium-id": MockModelEnum.STANDARD_MEDIUM_ID,
            "slow-high-id": MockModelEnum.SLOW_HIGH_ID,
            "local-model-id": MockModelEnum.LOCAL_MODEL_ID,
        }
        
        estimated_tokens = (100, 50)
        
        # Recalculated costs: fast-low=20, std-med=100, slow-high=200, local=0
        
        # Case 1: Default CHAT (Med/Std=100). Limit = 50. 
        # Initial filter selects only std-med. Cost limit filters it out.
        # Expect Error.
        with pytest.raises(AISetupError, match="No models found within cost limit"):
            # This call should raise the error, not return a model
            selector.select_model(
                use_case=UseCase.CHAT, max_cost=50, estimated_tokens=estimated_tokens
            )
            
        # Case 2: Default CODING (High/Slow=200). Limit = 150. 
        # Initial filter selects High/Slow. Cost limit filters it out.
        # Expect Error.
        with pytest.raises(AISetupError, match="No models found within cost limit"):
             # This call should also raise the error
            selector.select_model(
                use_case=UseCase.CODING, max_cost=150, estimated_tokens=estimated_tokens
            )

        # Patch ModelEnum only for the case that should succeed
        with patch('src.core.model_selector.ModelEnum', MockModelEnum):
            # Case 3: Low limit = 10. Only local (0) qualifies.
            selected_local_cost = selector.select_model(
                use_case=UseCase.DATA_ANALYSIS, max_cost=10, estimated_tokens=estimated_tokens
            )
            assert selected_local_cost == MockModelEnum.LOCAL_MODEL_ID
        
        # Case 4: Cost limit excludes all remains the same (error case)
        with pytest.raises(AISetupError, match="No models found within cost limit"):
             # Restore the call that is expected to raise the error
             selector.select_model(
                 use_case=UseCase.CHAT,
                 max_cost=1.0, # Below min cost of fast-low for (5,5)
                 estimated_tokens=(5,5)
             )

    def test_select_model_no_match_found(self, mock_unified_config):
        """Test select_model raises AISetupError when no suitable model is found initially."""
        selector = ModelSelector()
        # Use MockModelEnum here
        selector._model_id_to_enum = { 
             "fast-low-id": MockModelEnum.FAST_LOW_ID,
             "standard-medium-id": MockModelEnum.STANDARD_MEDIUM_ID,
             "slow-high-id": MockModelEnum.SLOW_HIGH_ID,
             "local-model-id": MockModelEnum.LOCAL_MODEL_ID,
        }

        # Case 1: Request parameters that don't match any model in DUMMY_MODELS_CONFIG
        # e.g., HIGH quality and LOCAL privacy (no model has both)
        with pytest.raises(AISetupError, match="No suitable model found for use case"):
            selector.select_model(
                use_case=UseCase.CHAT, # Use case doesn't matter as much as params
                quality="HIGH",
                privacy="LOCAL"
            )
            
        # Case 2: Request parameters for a non-existent quality/speed level
        with pytest.raises(AISetupError, match="No suitable model found for use case"):
            selector.select_model(
                use_case=UseCase.CODING, 
                quality="ULTRA_HIGH", # Not a defined quality in dummy data
                speed="INSTANT"      # Not a defined speed
            )

    def test_select_model_unknown_model_id_in_config(self, mock_unified_config):
         """Test select_model raises AISetupError if the selected model_id isn't in ModelEnum."""
         # selector = ModelSelector() # Don't initialize here, do it inside patch context
         
         # Create a deep copy of the config to modify locally
         import copy
         local_dummy_config = copy.deepcopy(DUMMY_MODELS_CONFIG)
         local_dummy_config["model-std-med"]["model_id"] = "unknown-id-for-testing"
         
         # Patch get_instance to return a mock using the LOCAL config copy
         with patch('src.core.model_selector.UnifiedConfig.get_instance') as mock_get_instance_local:
            mock_instance_local = MagicMock(spec=UnifiedConfig)
            mock_instance_local.get_all_models.return_value = local_dummy_config 
            # Explicitly mock return value for 'chat' with a NEW dictionary literal
            mock_instance_local.get_use_case_config.return_value = {
                "quality": "MEDIUM",
                "speed": "STANDARD",
                "privacy": "EXTERNAL"
            }
            mock_get_instance_local.return_value = mock_instance_local
            
            selector = ModelSelector() # Initialize selector *inside* patch context

            # Verify the mock setup (optional sanity check)
            assert selector._config.get_use_case_config('chat') == {"quality": "MEDIUM", "speed": "STANDARD", "privacy": "EXTERNAL"}

            # Expect the intended error about the missing enum
            with pytest.raises(AISetupError, match="No matching Model enum found for model_id"):
                selector.select_model(use_case=UseCase.CHAT)
         # No cleanup needed as the global DUMMY_MODELS_CONFIG was not modified
         # DUMMY_MODELS_CONFIG["model-std-med"] = original_model_config # Remove cleanup

    # --- Tests for internal methods (Optional but potentially useful) ---

    def test_filter_models(self, mock_unified_config):
        """Test the internal _filter_models logic."""
        selector = ModelSelector()
        all_models = DUMMY_MODELS_CONFIG
        
        # Case 1: Filter for MEDIUM quality, STANDARD speed
        params_med_std = {"quality": "MEDIUM", "speed": "STANDARD"}
        filtered_med_std = selector._filter_models(all_models, params_med_std)
        assert len(filtered_med_std) == 1
        assert filtered_med_std[0]["model_id"] == "standard-medium-id"
        
        # Case 2: Filter for LOW quality, FAST speed
        params_low_fast = {"quality": "LOW", "speed": "FAST"}
        filtered_low_fast = selector._filter_models(all_models, params_low_fast)
        assert len(filtered_low_fast) == 1
        assert filtered_low_fast[0]["model_id"] == "fast-low-id"
        
        # Case 3: Filter including privacy LOCAL
        params_med_fast_local = {"quality": "MEDIUM", "speed": "FAST", "privacy": "LOCAL"}
        filtered_local = selector._filter_models(all_models, params_med_fast_local)
        assert len(filtered_local) == 1
        assert filtered_local[0]["model_id"] == "local-model-id"

        # Case 4: Filter including privacy EXTERNAL (should match multiple)
        params_med_std_external = {"quality": "MEDIUM", "speed": "STANDARD", "privacy": "EXTERNAL"}
        filtered_external_med_std = selector._filter_models(all_models, params_med_std_external)
        assert len(filtered_external_med_std) == 1 # Only one matches MEDIUM/STANDARD/EXTERNAL
        assert filtered_external_med_std[0]["model_id"] == "standard-medium-id"
        
        # Case 5: Filter where privacy is NOT specified in params (should ignore privacy attribute)
        params_fast = {"quality": "LOW", "speed": "FAST"} # fast-low-id is EXTERNAL
        filtered_no_privacy_pref = selector._filter_models(all_models, params_fast)
        assert len(filtered_no_privacy_pref) == 1 
        assert filtered_no_privacy_pref[0]["model_id"] == "fast-low-id"
        
        # Case 6: No matching models
        params_no_match = {"quality": "HIGH", "speed": "FAST"}
        filtered_no_match = selector._filter_models(all_models, params_no_match)
        assert len(filtered_no_match) == 0

    def test_apply_cost_constraints(self, mock_unified_config):
        """Test the internal _apply_cost_constraints logic."""
        selector = ModelSelector()
        candidate_models = [
            DUMMY_MODELS_CONFIG["model-fast-low"],   # Cost: 20.0 min 0.01
            DUMMY_MODELS_CONFIG["model-std-med"],   # Cost: 100.0 min 0.05
            DUMMY_MODELS_CONFIG["model-slow-high"], # Cost: 200.0 min 0.1
            DUMMY_MODELS_CONFIG["model-local"]      # Cost: 0
        ]
        
        estimated_tokens = (100, 50)
        # Recalculated costs based on direct multiplication
        
        # Case 1: High cost limit (500), all pass
        max_cost_high = 500.0
        affordable_high = selector._apply_cost_constraints(candidate_models, max_cost_high, estimated_tokens)
        assert len(affordable_high) == 4
        assert {m['model_id'] for m in affordable_high} == {"fast-low-id", "standard-medium-id", "slow-high-id", "local-model-id"}
        
        # Case 2: Medium cost limit (150), excludes slow-high (200)
        max_cost_medium = 150.0 
        affordable_medium = selector._apply_cost_constraints(candidate_models, max_cost_medium, estimated_tokens)
        assert len(affordable_medium) == 3 
        assert {m['model_id'] for m in affordable_medium} == {"fast-low-id", "standard-medium-id", "local-model-id"}

        # Case 3: Low cost limit (50), excludes std-med (100) and slow-high (200)
        max_cost_low = 50.0
        affordable_low = selector._apply_cost_constraints(candidate_models, max_cost_low, estimated_tokens)
        assert len(affordable_low) == 2 
        assert {m['model_id'] for m in affordable_low} == {"fast-low-id", "local-model-id"}

        # Case 4: Very low cost limit (10), excludes fast-low (20) too
        max_cost_very_low = 10.0
        affordable_very_low = selector._apply_cost_constraints(candidate_models, max_cost_very_low, estimated_tokens)
        assert len(affordable_very_low) == 1 
        assert affordable_very_low[0]["model_id"] == "local-model-id"
        
        # Case 5: Zero cost limit, only local passes
        max_cost_zero = 0.0
        affordable_zero = selector._apply_cost_constraints(candidate_models, max_cost_zero, estimated_tokens)
        assert len(affordable_zero) == 1
        assert affordable_zero[0]["model_id"] == "local-model-id"
        
        # Case 6: Minimum cost check
        estimated_tokens_min = (5, 5)
        # fast-low calc = 0.1*5 + 0.2*5 = 1.5. Min cost is 0.01. Calculated cost is used.
        max_cost_below_min = 1.0 # Limit is below calculated cost (1.5)
        affordable_below_min = selector._apply_cost_constraints(candidate_models, max_cost_below_min, estimated_tokens_min)
        assert len(affordable_below_min) == 1 # fast-low costs 1.5, too high. Only local (0) remains.
        assert affordable_below_min[0]["model_id"] == "local-model-id"

        # Case 7: Test with a model missing cost info (remains the same)
        candidate_with_missing_cost = candidate_models + [{
            "model_id": "no-cost-info-id", 
            "provider": "prov-X",
            "quality": "MEDIUM",
            "speed": "STANDARD"
            # Missing 'cost' key
        }]
        affordable_missing = selector._apply_cost_constraints(candidate_with_missing_cost, max_cost_high, estimated_tokens)
        assert len(affordable_missing) == 4 # Excludes the one without cost info
        assert "no-cost-info-id" not in {m['model_id'] for m in affordable_missing}

    def test_select_best_model(self, mock_unified_config):
        """Test the internal _select_best_model logic."""
        selector = ModelSelector()
        params = {"quality": "HIGH", "speed": "FAST"} # Example params, may not be used directly by _select_best_model
        
        # Case 1: Clear winner (High/Fast)
        candidates1 = [
            DUMMY_MODELS_CONFIG["model-fast-low"],   # Low/Fast
            DUMMY_MODELS_CONFIG["model-std-med"],   # Med/Std
            DUMMY_MODELS_CONFIG["model-slow-high"], # High/Slow
            # Add a hypothetical High/Fast model
            {"model_id": "high-fast-id", "quality": "HIGH", "speed": "FAST"}
        ]
        best_model_id1 = selector._select_best_model(candidates1, params)
        assert best_model_id1 == "high-fast-id"
        
        # Case 2: Tie in quality, speed breaks tie (both HIGH, prefer FAST)
        candidates2 = [
            DUMMY_MODELS_CONFIG["model-slow-high"], # High/Slow
            # Add a hypothetical High/Fast model
            {"model_id": "high-fast-id", "quality": "HIGH", "speed": "FAST"}
        ]
        best_model_id2 = selector._select_best_model(candidates2, params)
        assert best_model_id2 == "high-fast-id"
        
        # Case 3: Tie in speed, quality breaks tie (both FAST, prefer HIGH)
        candidates3 = [
            DUMMY_MODELS_CONFIG["model-fast-low"],   # Low/Fast
            # Add a hypothetical High/Fast model
            {"model_id": "high-fast-id", "quality": "HIGH", "speed": "FAST"}
        ]
        best_model_id3 = selector._select_best_model(candidates3, params)
        assert best_model_id3 == "high-fast-id"
        
        # Case 4: Only one candidate
        candidates4 = [DUMMY_MODELS_CONFIG["model-std-med"]]
        best_model_id4 = selector._select_best_model(candidates4, params)
        assert best_model_id4 == "standard-medium-id"
        
        # Case 5: Models with missing quality/speed (should use defaults/lower weight)
        candidates5 = [
            {"model_id": "missing-qs-id"}, # No quality/speed -> treated as MEDIUM/STANDARD (weight 2/2)
            DUMMY_MODELS_CONFIG["model-fast-low"], # Low/Fast (weight 1/3)
        ]
        best_model_id5 = selector._select_best_model(candidates5, params)
        # Missing Q/S (2,2) should beat Low/Fast (1,3) based on primary sort key (quality)
        assert best_model_id5 == "missing-qs-id"
        
    def test_build_model_mapping(self, mock_unified_config):
         """Test the internal _build_model_mapping logic."""
         
         # Mock ModelEnum for predictable testing
         # Use names matching expected mapping results
         class MockModelEnum(Enum):
             FAST_LOW_ID = "fast-low-id" # Matches model_id directly
             # Intentionally omit direct match for std-med to test fallback
             MODEL_STD_MED = "standard-medium-id-fallback" # Fallback finds this via key 'model-std-med'
             SLOW_HIGH_ID = "slow-high-id"
             LOCAL_MODEL_ID = "local-model-id" # Not used in this specific test data, but keep for context
             SOME_OTHER_MODEL = "other-model-value"
             
         # Test data tailored for mapping logic
         test_config_data: Dict[str, Dict[str, Any]] = {
             "model-fast-low": { # Exact model_id match -> FAST_LOW_ID
                 "model_id": "fast-low-id", "provider": "prov-A",
             },
             "model-std-med": { # Fallback via key match -> MODEL_STD_MED
                 "model_id": "standard-medium-id-fallback", # Doesn't match any enum value directly
                 "provider": "prov-B",
             },
             "model-slow-high": { # Exact model_id match -> SLOW_HIGH_ID
                 "model_id": "slow-high-id", "provider": "prov-C",
             },
             "model-no-match": { # Neither model_id nor key match enum -> Not in mapping
                 "model_id": "no-matching-enum-value", "provider": "prov-D",
             },
             "model-missing-id": { # Missing model_id key -> Not in mapping
                 "provider": "prov-E",
             }
         }
         
         # Patch UnifiedConfig specifically for this test to return the test data
         # Also patch ModelEnum within the selector's scope
         with patch('src.core.model_selector.UnifiedConfig.get_instance') as mock_get_instance_local, \
              patch('src.core.model_selector.ModelEnum', MockModelEnum): # Patch the Enum used by the selector
            mock_instance_local = MagicMock(spec=UnifiedConfig)
            mock_instance_local.get_all_models.return_value = test_config_data
            mock_get_instance_local.return_value = mock_instance_local
            
            # Initialize selector *inside* the patch context
            selector = ModelSelector()
            mapping = selector._model_id_to_enum

            # Assertions
            assert len(mapping) == 3 # Only 3 models should successfully map
            
            # Check exact match case
            assert "fast-low-id" in mapping
            assert mapping["fast-low-id"] == MockModelEnum.FAST_LOW_ID
            
            # Check fallback match case
            assert "standard-medium-id-fallback" in mapping
            assert mapping["standard-medium-id-fallback"] == MockModelEnum.MODEL_STD_MED
            
            # Check another exact match case
            assert "slow-high-id" in mapping
            assert mapping["slow-high-id"] == MockModelEnum.SLOW_HIGH_ID
            
            # Check cases that should NOT be in the mapping
            assert "no-matching-enum-value" not in mapping
            # Ensure keys related to models without model_id aren't added
            assert None not in mapping 
            assert "model-missing-id" not in mapping # Check key isn't used if model_id missing