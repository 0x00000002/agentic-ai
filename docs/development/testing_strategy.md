# Comprehensive Testing Strategy

This document outlines the general testing strategy for the Agentic-AI framework, covering unit, integration, and potentially end-to-end testing approaches. It serves as the primary guide for writing tests.

**Note:**

- Provider-specific testing details are found in `docs/development/testing_providers.md`.
- For other complex modules (e.g., specific agents, core AI classes), detailed testing guidance may be included as a "Testing" subsection within their primary documentation page.

## I. Overarching Principles

1.  **Testing Pyramid:** Adhere to the testing pyramid principle:
    - **Base:** Comprehensive **Unit Tests** for individual components. Focus on testing logic in isolation.
    - **Middle:** Focused **Integration Tests** for interactions between key components. Verify contracts and collaborations.
    - **Top:** Minimal **End-to-End (E2E) Tests** for critical user flows (if applicable, especially involving the UI). Ensure the system works from user input to final output.
2.  **Test Granularity:** Write tests at the appropriate level. Unit tests should mock dependencies outside the unit (e.g., external APIs, filesystem, other classes), while integration tests should involve real instances of collaborating components (mocking only external systems like actual LLM APIs or databases not under test).
3.  **Framework:** Use `pytest` as the primary testing framework. Leverage fixtures (`tests/conftest.py`) for reusable setup (mocks, configurations, test data, temporary files/directories).
4.  **Mocking:** Use `unittest.mock` (`Mock`, `MagicMock`, `patch`) for mocking dependencies in unit tests. Keep mocks focused on the interaction boundary – mock _what_ a dependency returns or _that_ it was called correctly, not its internal implementation.
5.  **Coverage:** Aim for high unit test coverage on core logic, complex algorithms, and critical components. Use coverage reports (`pytest-cov`) as a guide but prioritize testing essential behaviors, edge cases, and error conditions over blindly chasing 100% line coverage.
6.  **Location:** Maintain a clear and consistent test structure mirroring the `src/` directory:
    - `tests/unit/<component>/`: For unit tests corresponding to `src/<component>/`.
    - `tests/integration/<component_or_workflow>/`: For integration tests focusing on interactions between specific components or end-to-end workflows within the backend.
    - `tests/e2e/`: For true end-to-end tests involving external interfaces like a UI or API gateway (if applicable).
7.  **Verify, don't assume:** When creating tests for any class, verify the constructor's arguments, class methods, and their arguments. Don't assume you know them, as the consumer class may overlook changes made in the class it uses after some refactoring.

## II. Component-Specific Strategies

The following outlines the recommended testing approach for each major component area found in `src/`.

1.  **`config/` (`UnifiedConfig`, Loaders, etc.):**
    - **Unit Tests:** High priority. Test loading from default paths, specified paths, environment variables. Verify correct merging logic (e.g., file overrides defaults, user overrides base). Test access methods (`get_provider_config`, `get_model_config`, `get_api_key`, etc.). Test error handling for missing files, malformed content (e.g., invalid YAML/JSON), and missing required keys. Mock filesystem operations (`pathlib.Path`, `open`, `os.environ`) extensively using fixtures like `tmp_path`.
2.  **`core/`:**
    - **`providers/`:** Follow the detailed strategy in `docs/development/testing_providers.md`. Focus on unit tests mocking the specific provider SDKs or HTTP clients (`openai`, `anthropic`, `google.generativeai`, `ollama`, `requests`).
    - **`interfaces/`:** No explicit tests needed (abstract definitions).
    - **`models/`:** Unit tests for any custom validation logic, methods, or complex default factories within Pydantic/dataclass models. Basic validation is implicitly tested during usage in other components.
    - **`provider_factory.py`:** Unit tests covering provider registration (including duplicates/errors), creation logic (correct class instantiation, parameter passing from config), retrieval of registered providers, and error handling for unknown providers or configuration issues. Mock `UnifiedConfig`.
    - **`base_ai.py` / `tool_enabled_ai.py` (Main AI Classes):**
      - **Unit Tests:** Test core logic like prompt assembly (if complex), basic provider interaction (using a mock `ProviderInterface`), handling simple provider responses (`ProviderResponse`), parsing tool calls, invoking the tool manager, and formatting tool results for the provider. Mock `ProviderFactory` or `ProviderInterface`, `ToolManager`, `ConversationManager`.
      - **Integration Tests:** Test interaction with a _real_ `ProviderFactory` (using mock providers). Test the full request->tool_call->tool_result->request loop using mock providers that simulate tool calls and responses, interacting with a real `ConversationManager` and `ToolManager` (using mock tools/executor). Test error handling during the loop (provider errors, tool errors).
    - **`model_selector.py`:** Unit tests covering model selection logic based on use case, quality, speed, cost constraints. Test filtering, cost calculation, best model selection, and enum mapping. Mock `UnifiedConfig` to provide controlled model/use-case data.
3.  **`tools/`:**
    - **`tool_registry.py`:** Unit tests for registering tools (checking for duplicates, validation), formatting tools for different provider types (OpenAI, Anthropic, etc.), retrieving tool definitions.
    - **`tool_executor.py`:** Unit tests for executing tool functions successfully, handling execution errors (capturing exceptions in `ToolResult`), enforcing timeouts (patching `signal` or using appropriate async constructs if refactored), implementing retries. Mock the actual tool functions being executed.
    - **`tool_manager.py`:** Unit tests for coordinating registration (via `ToolRegistry`), execution (via `ToolExecutor`), and potentially formatting. Mock `ToolRegistry` and `ToolExecutor`.
    - **Specific Tools (functions/classes defined):** Unit test the logic of each individual tool function/class itself, mocking any external dependencies (APIs, libraries) they might use.
4.  **`agents/`:**
    - **`base_agent.py`:** Unit tests for any common setup, helper methods, or abstract logic defined in the base class.
    - **`agent_registry.py`:** Unit tests covering class registration (mocking `issubclass`), handling duplicates (overwriting), retrieving classes, and handling invalid types. Mock internal `_register_agents` call during init for isolation.
    - **`agent_factory.py`:** Unit tests covering agent class registration, creation logic based on type and configuration, dependency injection (if applicable), retrieval of registered types, and error handling for unknown types. Mock `AgentRegistry` and the agent classes being instantiated.
    - **`agent_registrar.py`:** Unit tests verifying that the correct agent classes are registered with the provided registry mock.
    - **Specific Agents (`Coordinator`, `RequestAnalyzer`, etc.):**
      - **Unit Tests:** Test the agent's specific logic, decision-making processes, state management, and interaction with its direct dependencies. Mock dependencies like other agents, `ToolEnabledAI` (or its interface), `ToolManager`, `PromptTemplate`, etc. Test different input scenarios and expected outputs or state changes.
      - **Integration Tests:** Test interactions between collaborating agents (e.g., `Coordinator` -> `RequestAnalyzer` -> `SpecificTaskAgent`). Test agents interacting with a _real_ (but potentially configured with mock providers) `ToolEnabledAI` instance to verify the flow of data and control.
5.  **`prompts/`:**
    - **Unit Tests:** Test template loading (from file or string), rendering with various valid/invalid inputs (including missing variables), handling different template formats if supported, and potential error conditions during rendering.
6.  **`conversation/`:**
    - **Unit Tests:** Test conversation history management: adding user/assistant/tool messages, retrieving history (full or truncated), enforcing length limits (token count or message count), formatting history for different provider needs, serialization/deserialization if applicable.
7.  **`utils/`:**
    - **Unit Tests:** Test each utility function or class independently. Ensure pure functions are tested with various inputs and edge cases. Mock external dependencies for utilities that interact with I/O, network, etc. (e.g., file system wrappers, HTTP clients, logger backends).
8.  **`metrics/`:**
    - **Unit Tests:** Test metric collection logic (incrementing counters, recording timings), aggregation methods, and formatting/reporting logic. Mock the underlying storage or reporting mechanism (e.g., logging, database, external monitoring service).
9.  **`exceptions.py`:** No explicit tests needed (definitions only). Their correct usage and propagation are tested in other components.
10. **`ui/`:** (If a UI component exists)
    - **Unit Tests:** For any backend API handlers, data processing logic, or state management associated specifically with the UI. Mock interactions with other backend components (`agents`, `core`, etc.).
    - **E2E Tests:** Use browser automation tools (e.g., Playwright, Selenium) to simulate user interactions and verify end-to-end flows through the UI and backend. Focus on critical paths.

## III. Integration Test Priorities

Start with integration tests that cover fundamental interactions:

1.  Config loading -> Provider Factory -> Provider Creation (using mock SDKs).
2.  Core AI (`ToolEnabledAI`) -> Provider Interaction (using mock `ProviderInterface` simulating success, errors, tool calls).
3.  Core AI (`ToolEnabledAI`) -> Tool Manager -> Tool Executor (using mock tool functions).
4.  Full Tool Loop: Core AI -> Mock Provider (returns tool call) -> Tool Manager -> Mock Tool -> Core AI -> Mock Provider (processes tool result).
5.  Agent (`Coordinator`) -> Core AI (`ToolEnabledAI`) interaction (basic request/response).
6.  Agent (`Coordinator`) -> Core AI (`ToolEnabledAI`) -> Tool loop execution.

## IV. Test Execution and CI

- Tests should be easily runnable locally via `pytest`.
- Integrate test execution into the Continuous Integration (CI) pipeline (e.g., GitHub Actions).
- Run linters (e.g., Ruff, MyPy) and formatters (e.g., Black, isort) alongside tests in CI.
- Consider running tests automatically on pull requests.
- Track test coverage and identify significant gaps in critical areas.

## V. Prioritization

1.  **Unit Tests - Foundational:** `config`, `core/providers` (as per `testing_providers.md`), `tools` (core classes), `prompts`, `conversation`, `utils`.
2.  **Unit Tests - Core Logic:** `core/ai_core`, `agents` (base and specific).
3.  **Integration Tests:** Start with the priority list in Section III.
4.  **Unit Tests - Others:** `metrics`, `core/models` (if complex).
5.  **E2E Tests:** If UI exists.
