import asyncio
import json
import os
from typing import Dict, Any, Optional

from aiohttp import web


class MockMCPServer:
    """A mock MCP server for testing MCPClientManager interactions."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self._requested_port = port
        self._actual_port = None
        self.app = web.Application()
        self.app.router.add_get("/", self.handle_root)
        self.app.router.add_post("/call_tool", self.handle_call_tool)
        self._runner = None
        self._site = None
        # Define expected behavior (can be customized by tests)
        self.expected_tool_responses: Dict[str, Any] = {}
        self.expected_auth_token: Optional[str] = None
        self.simulate_server_error: bool = False
        self.simulate_connection_error: bool = False # Note: Difficult to simulate directly here

    async def handle_call_tool(self, request: web.Request) -> web.Response:
        """Handles incoming /call_tool requests."""
        if self.simulate_server_error:
            return web.Response(status=500, text="Simulated Internal Server Error")

        # Check Authentication
        if self.expected_auth_token:
            auth_header = request.headers.get("Authorization")
            if not auth_header or auth_header != f"Bearer {self.expected_auth_token}":
                return web.Response(status=401, text="Unauthorized")

        try:
            data = await request.json()
            tool_name = data.get("tool_name")
            # arguments = data.get("arguments", {}) # Arguments might be used later

            if tool_name in self.expected_tool_responses:
                response_data = self.expected_tool_responses[tool_name]
                # Allow simulating tool-specific errors vs general success
                if isinstance(response_data, dict) and response_data.get("error"):
                    return web.json_response(
                        {"error": response_data["error"], "error_details": response_data.get("error_details")},
                        status=response_data.get("status", 400) # Default to 400 for tool errors
                    )
                else:
                     # Assume success if not an error structure
                    return web.json_response({"result": response_data})
            else:
                # Tool not found in our mock configuration
                return web.json_response(
                    {"error": "ToolNotFound", "error_details": f"Tool '{tool_name}' not found on mock server."},
                    status=404
                )

        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON payload")
        except Exception as e:
            # Catch unexpected issues within the handler
            return web.Response(status=500, text=f"Mock server handler error: {str(e)}")

    async def handle_root(self, request: web.Request) -> web.Response:
        """Handles requests to the root path, mimicking an SSE stream endpoint."""
        # Use StreamResponse for SSE
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={'Content-Type': 'text/event-stream',
                     'Cache-Control': 'no-cache',
                     'Connection': 'keep-alive'}
        )
        await response.prepare(request)

        try:
            # Send an initial SSE comment to confirm connection
            await response.write(b": stream opened\n\n")
            
            # Keep the connection open, waiting for client disconnect
            # This allows the sse_reader task to potentially signal readiness
            while True:
                # Check if connection is closed by peer
                if request.transport is None or request.transport.is_closing():
                    break
                # Sleep briefly to avoid busy-waiting
                await asyncio.sleep(1)
                # Optionally send periodic keep-alive comments if needed
                # await response.write(b": keepalive\n\n")

        except asyncio.CancelledError:
            # Client disconnected or server stopping
            print("SSE stream handler cancelled.")
        except Exception as e:
            print(f"Error in SSE stream handler: {e}")
        finally:
            # Ensure the stream is properly ended
            # await response.write_eof() # Might not be needed/correct for SSE
            pass

        return response

    async def start(self):
        """Starts the mock server."""
        if self._runner or self._site:
            print("Mock server already running or not properly stopped.")
            return

        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self._requested_port)
        await self._site.start()
        # Get the actual port assigned by the OS
        if self._site._server is not None and self._site._server.sockets:
             # Accessing socket details might differ slightly based on aiohttp version
             # Common pattern: access the socket's address info
             addr = self._site._server.sockets[0].getsockname()
             self._actual_port = addr[1]
             print(f"Mock MCP Server started at http://{self.host}:{self._actual_port}")
        else:
             print(f"Mock MCP Server started, but could not determine dynamic port.")
             # Fallback or raise error if port is critical?
             # For now, let's assume it works or subsequent steps will fail clearly.
             self._actual_port = self._requested_port # Fallback to requested (might be 0)

    @property
    def port(self) -> Optional[int]:
        """Return the actual port the server is listening on."""
        return self._actual_port

    async def stop(self):
        """Stops the mock server."""
        if self._site:
            await self._site.stop()
            self._site = None
            print("Mock MCP Server site stopped.")
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            print("Mock MCP Server runner cleaned up.")

    # --- Configuration methods for tests ---
    def set_expected_token(self, token: Optional[str]):
        self.expected_auth_token = token

    def add_tool_response(self, tool_name: str, response: Any):
        self.expected_tool_responses[tool_name] = response

    def set_simulate_server_error(self, simulate: bool):
        self.simulate_server_error = simulate

    def reset(self):
        """Resets mock server state between tests."""
        self.expected_tool_responses = {}
        self.expected_auth_token = None
        self.simulate_server_error = False


# Example of running the server directly (for testing the mock itself)
async def main():
    server = MockMCPServer()
    # Configure for direct testing
    server.set_expected_token("test-token-123")
    server.add_tool_response("mock_tool_success", {"data": "Success!"})
    server.add_tool_response("mock_tool_error", {"error": "ToolSpecificError", "error_details": "Something went wrong internally.", "status": 500})
    server.add_tool_response("mock_auth_tool", {"data": "Authenticated access granted."})


    await server.start()
    print("Server running. Press Ctrl+C to stop.")
    try:
        # Keep server running until interrupted
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("Stopping server...")
    finally:
        await server.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
