import asyncio
import os
import tempfile
import unittest
import yaml
from typing import Dict, Any
import logging # Import standard logging
import sys # Import sys to get stdout for StreamHandler

# Import the mock server
from tests.mocks.mock_mcp_server import MockMCPServer

# Import necessary components from the framework
# (Adjust paths based on your project structure if needed)
from src.config.unified_config import UnifiedConfig
from src.mcp.mcp_client_manager import MCPClientManager
from src.tools.tool_manager import ToolManager
from src.tools.models import ToolCall, ToolDefinition, ToolResult
from src.utils.logger import LoggerFactory, LoggerInterface, LoggingLevel # Import logger components


# Define a known port for the mock server
MOCK_SERVER_HOST = "127.0.0.1"
# MOCK_SERVER_PORT = 8085 # No longer fixed
# MOCK_SERVER_URL = f"http://{MOCK_SERVER_HOST}:{MOCK_SERVER_PORT}" # Constructed dynamically

# Define a test auth token and env var name
TEST_AUTH_TOKEN = "test-secret-token-xyz"
TEST_AUTH_ENV_VAR = "MCP_TEST_AUTH_TOKEN"


class TestMCPIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for MCP client functionality using a mock server."""

    mock_server: MockMCPServer
    temp_dir: tempfile.TemporaryDirectory
    mcp_config_path: str
    original_config_instance: Any # To restore later
    original_env: Dict[str, str]

    @classmethod
    def setUpClass(cls):
        # Store original env vars to restore later
        cls.original_env = os.environ.copy()

    @classmethod
    def tearDownClass(cls):
        # Restore original env vars
        os.environ.clear()
        os.environ.update(cls.original_env)

    async def asyncSetUp(self):
        """Set up the test environment before each test."""
        # Ensure real loggers are used for this test
        LoggerFactory.enable_real_loggers()
        self.addAsyncCleanup(LoggerFactory.disable_real_loggers)

        # 1. Start the Mock Server (port=0 for dynamic assignment)
        self.mock_server = MockMCPServer(host=MOCK_SERVER_HOST, port=0)
        await self.mock_server.start()
        # Retrieve the dynamically assigned port
        actual_port = self.mock_server.port
        self.assertIsNotNone(actual_port, "Mock server did not report an actual port after starting.")
        self.assertNotEqual(actual_port, 0, "Mock server port is still 0 after starting.")
        MOCK_SERVER_URL = f"http://{MOCK_SERVER_HOST}:{actual_port}"
        print(f"--- Test using Mock Server URL: {MOCK_SERVER_URL} ---") # Log for debugging

        self.addAsyncCleanup(self.mock_server.stop) # Ensure server stops even if setup fails later

        # 2. Create Temporary Config Directory and mcp.yml (using dynamic URL)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup) # Ensure directory is cleaned up

        self.mcp_config_path = os.path.join(self.temp_dir.name, "mcp.yml")
        test_mcp_config = {
            "mcp_servers": {
                "mock_server_1": {
                    "description": "Primary mock server",
                    "url": MOCK_SERVER_URL,
                    "provides_tools": [
                        {
                            "name": "mock_tool_no_auth",
                            "description": "A simple mock tool.",
                            "inputSchema": {"type": "object", "properties": {"param": {"type": "string"}}},
                            "required": ["param"]
                        }
                    ]
                },
                "mock_server_auth": {
                    "description": "Mock server requiring auth",
                    "url": MOCK_SERVER_URL, # Points to the same mock server instance
                    "auth": {
                        "type": "bearer",
                        "token_env_var": TEST_AUTH_ENV_VAR
                    },
                    "provides_tools": [
                        {
                            "name": "mock_tool_with_auth",
                            "description": "A mock tool requiring authentication.",
                            "inputSchema": {"type": "object", "properties": {"data": {"type": "integer"}}},
                            "required": ["data"]
                        }
                    ]
                }
            }
        }
        with open(self.mcp_config_path, 'w') as f:
            yaml.dump(test_mcp_config, f)

        # 3. Set Auth Environment Variable
        os.environ[TEST_AUTH_ENV_VAR] = TEST_AUTH_TOKEN

        # 4. Configure UnifiedConfig to use the temporary config
        # Reset singleton and configure with only the mock mcp.yml
        self.original_config_instance = UnifiedConfig._instances.get(UnifiedConfig)
        UnifiedConfig.reset_instance() # Use the correct reset method

        # We need to simulate having a base config directory even if tools.yml is empty
        # Create a dummy tools.yml as well
        dummy_tools_path = os.path.join(self.temp_dir.name, "tools.yml")
        with open(dummy_tools_path, 'w') as f:
            yaml.dump({"tools": []}, f)

        # Get the config instance, pointing it to the temporary directory
        # This automatically loads mcp.yml and tools.yml from temp_dir
        self.config = UnifiedConfig.get_instance(config_dir=self.temp_dir.name)

        # Ensure config is reset after the test
        self.addAsyncCleanup(UnifiedConfig.reset_instance)


        # 5. Instantiate Managers (using the new config)
        # Use the factory now that real loggers are enabled
        self.test_logger = LoggerFactory.create(name="test_mcp_integration", level=LoggingLevel.DEBUG)
        # Add handler to see logs during test run (optional, good for debugging)
        # Check if handlers already exist to avoid duplicates if tests run weirdly
        # Note: Logger itself might handle adding handlers based on its config, 
        # but adding one ensures we see output if needed for debugging.
        # If LoggerFactory configures handlers, this might add a duplicate.
        if not self.test_logger._logger.handlers: # Access internal logger
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.test_logger._logger.addHandler(handler)

        # MCPClientManager needs config and our standard logger
        self.mcp_manager = MCPClientManager(config=self.config, logger=self.test_logger)
        self.addAsyncCleanup(self.mcp_manager.close_all_clients)

        # ToolManager needs config, our standard logger, and the injected mcp_manager
        self.tool_manager = ToolManager(
            unified_config=self.config, 
            logger=self.test_logger,
            mcp_client_manager=self.mcp_manager
        )


    async def reset_unified_config(self):
        """Resets UnifiedConfig singleton after test."""
        UnifiedConfig.reset_instance() # Use the correct reset method
        # Restore original instance if it existed
        if self.original_config_instance:
             UnifiedConfig._instances[UnifiedConfig] = self.original_config_instance


    async def asyncTearDown(self):
        """Clean up after each test."""
        # LoggerFactory cleanup is handled by addAsyncCleanup
        # Mock server stop and temp dir cleanup are handled by addAsyncCleanup/addCleanup
        # Unset the environment variable
        if TEST_AUTH_ENV_VAR in os.environ:
            del os.environ[TEST_AUTH_ENV_VAR]
        # Resetting UnifiedConfig is handled by addAsyncCleanup


    # --- Test Cases ---

    async def test_successful_mcp_call_no_auth(self):
        """Verify successful execution of a non-authenticated MCP tool."""
        tool_name = "mock_tool_no_auth"
        arguments = {"param": "test_value"}
        expected_result = {"status": "ok", "data": "processed: test_value"}

        # Configure the mock server response for this specific tool
        self.mock_server.reset() # Ensure clean state
        self.mock_server.add_tool_response(tool_name, expected_result)

        # Create the ToolCall object
        tool_call = ToolCall(id="call_123", name=tool_name, arguments=arguments)

        # Execute the tool via ToolManager
        result: ToolResult = await self.tool_manager.execute_tool(tool_call)

        # Assertions
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertEqual(result.result, expected_result)
        self.assertEqual(result.tool_name, tool_name)
        self.assertEqual(result.metadata.get('original_call_id'), "call_123")

    async def test_successful_mcp_call_with_auth(self):
        """Verify successful execution of an authenticated MCP tool."""
        tool_name = "mock_tool_with_auth"
        arguments = {"data": 42}
        expected_result = {"message": "Auth successful!", "value": 42 * 2}

        # Configure the mock server response and required auth
        self.mock_server.reset()
        self.mock_server.set_expected_token(TEST_AUTH_TOKEN)
        self.mock_server.add_tool_response(tool_name, expected_result)

        # Create the ToolCall object
        tool_call = ToolCall(id="call_456", name=tool_name, arguments=arguments)

        # Execute the tool via ToolManager
        result: ToolResult = await self.tool_manager.execute_tool(tool_call)

        # Assertions
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertEqual(result.result, expected_result)
        self.assertEqual(result.tool_name, tool_name)
        self.assertEqual(result.metadata.get('original_call_id'), "call_456")

    async def test_mcp_call_auth_failure_wrong_token(self):
        """Verify auth failure when the client provides the wrong token (mock expects different)."""
        tool_name = "mock_tool_with_auth"
        arguments = {"data": 99}

        # Configure the mock server to expect a DIFFERENT token
        self.mock_server.reset()
        self.mock_server.set_expected_token("wrong-token-789") # Mock expects this
        # We don't need to add a success response as auth should fail first

        # Create the ToolCall object
        tool_call = ToolCall(id="call_789", name=tool_name, arguments=arguments)

        # Execute the tool via ToolManager (Client will send TEST_AUTH_TOKEN)
        result: ToolResult = await self.tool_manager.execute_tool(tool_call)

        # Assertions - Expecting a failure result
        self.assertIsNotNone(result)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertIn("HTTP Error 401", result.error) # Check actual error from wrapper
        self.assertIn("401", result.error)      # Status code should be in the error
        self.assertIn("Unauthorized", result.error)
        self.assertIsNone(result.result)
        self.assertEqual(result.tool_name, tool_name)
        self.assertEqual(result.metadata.get('original_call_id'), "call_789")

    async def test_mcp_call_tool_not_found_on_server(self):
        """Verify failure when the tool exists in config but not on the mock server."""
        tool_name = "mock_tool_no_auth" # This tool is in our test mcp.yml
        arguments = {"param": "irrelevant"}

        # Configure the mock server - crucial part is NOT adding a response for tool_name
        self.mock_server.reset()
        # self.mock_server.add_tool_response(tool_name, ...) # <--- Intentionally omitted

        # Create the ToolCall object
        tool_call = ToolCall(id="call_abc", name=tool_name, arguments=arguments)

        # Execute the tool via ToolManager
        result: ToolResult = await self.tool_manager.execute_tool(tool_call)

        # Assertions - Expecting a failure result
        self.assertIsNotNone(result)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertIn("ToolNotFound", result.error) # Check specific error from JSON body
        # Check for the specific error text returned by the mock server's /call_tool handler
        self.assertIn("Tool 'mock_tool_no_auth' not found on mock server", result.error)
        self.assertIsNone(result.result)
        self.assertEqual(result.tool_name, tool_name)
        self.assertEqual(result.metadata.get('original_call_id'), "call_abc")

    async def test_mcp_call_server_error(self):
        """Verify failure when the mock server returns a 500 error."""
        tool_name = "mock_tool_no_auth" # A valid tool in config
        arguments = {"param": "trigger_server_error"}

        # Configure the mock server to simulate a 500 error
        self.mock_server.reset()
        self.mock_server.set_simulate_server_error(True)

        # Create the ToolCall object
        tool_call = ToolCall(id="call_xyz", name=tool_name, arguments=arguments)

        # Execute the tool via ToolManager
        result: ToolResult = await self.tool_manager.execute_tool(tool_call)

        # Assertions - Expecting a failure result
        self.assertIsNotNone(result)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertIn("HTTP Error 500", result.error) # Check actual error from wrapper
        self.assertIn("500", result.error)      # Status code
        self.assertIn("Server Error", result.error) # Check for part of the mock's text
        self.assertIsNone(result.result)
        self.assertEqual(result.tool_name, tool_name)
        self.assertEqual(result.metadata.get('original_call_id'), "call_xyz")

    async def test_mcp_call_connection_error(self):
        """Verify failure when the MCP server is unreachable."""
        tool_name = "mock_tool_no_auth" # A valid tool in config
        arguments = {"param": "no_connection"}

        # Stop the mock server BEFORE the call
        await self.mock_server.stop()

        # Create the ToolCall object
        tool_call = ToolCall(id="call_fail", name=tool_name, arguments=arguments)

        # Execute the tool via ToolManager
        result: ToolResult = await self.tool_manager.execute_tool(tool_call)

        # Assertions - Expecting a failure result due to connection error
        self.assertIsNotNone(result)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertIn("Connection Error", result.error) # Check actual error from wrapper
        # Check for indicators of connection failure (aiohttp/os specific)
        # Examples: 'Connection refused', 'Cannot connect to host'
        self.assertTrue(
            'Connection refused' in result.error or
            'Cannot connect to host' in result.error or
            'Connection error' in result.error # Generic fallback
        )
        self.assertIsNone(result.result)
        self.assertEqual(result.tool_name, tool_name)
        self.assertEqual(result.metadata.get('original_call_id'), "call_fail")

    # Add more specific tests below...


if __name__ == '__main__':
    unittest.main()
