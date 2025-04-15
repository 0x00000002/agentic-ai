# Development Plan: Personal AI Assistant Framework

This document outlines the planned phases for developing the Agentic-AI framework into a practical, responsive personal AI assistant. The primary target deployment uses a cost-effective, always-on VM (<$50/month) to ensure responsiveness by eliminating cold starts for the core interaction loop. Serverless functions (e.g., AWS Lambda) can be utilized for specific tool backends where appropriate.

**Goal:** Create a practical, responsive personal AI assistant framework (initially integrated with Telegram) deployable primarily on a small, always-on VM, leveraging MCP for external tools (whose backends might be serverless).

**Note:** This plan focuses on the core Agentic-AI framework and its direct integrations. The implementation details of external MCP servers are considered separate projects.

**Overarching Considerations:**

- **Testing Strategy:** Each phase requires appropriate unit and integration tests (e.g., for ConversationStore implementations, Telegram integration logic, Core AI behavior with persistence).
- **Dependency Management:** New dependencies (`boto3`, `aiofiles`, `python-telegram-bot`, etc.) must be added to `requirements.txt` and managed.

## Phase 1: Core Framework - Persistence Layer

- **Objective:** Eliminate reliance on in-memory conversation history for robustness and long-running operation. Essential for both reliable VM operation and any potential serverless components.
- **Tasks:**
  1.  **Design `ConversationStore` Interface:**
      - Create `src/conversation/store/base_store.py` (`abc.ABC`).
      - Define methods: `async load_conversation(id)`, `async save_conversation(id, messages)`, `async delete_conversation(id)`.
  2.  **Implement `FileConversationStore`:**
      - Create `src/conversation/store/file_store.py`.
      - Store conversations as JSON/pickle in a configured directory.
      - Use `aiofiles` for async I/O.
  3.  **(Recommended) Implement `DynamoDBConversationStore`:**
      - Create `src/conversation/store/dynamodb_store.py`.
      - Use `boto3`.
      - Configure table name/region in `config.yml`.
      - Requires AWS credentials (accessible from VM via IAM Role or env vars).
      - **Attention:** Consider DynamoDB item size limits (400KB). Long conversations might require strategies like splitting history, compression, or storing messages individually.
      - **This is the preferred approach for robust VM deployment.**
  4.  **Integrate Store into `ConversationManager`:**
      - Modify `__init__` to accept `conversation_id` and `ConversationStore` instance.
      - Adapt methods (`add_message`, `get_messages`, `clear_messages`, `reset`) to use the store.
      - **Detail:** Define interaction pattern (e.g., load history on first `get_messages` or `add_message`, save on every `add_message`).
  5.  **Adapt Higher Layers:**
      - Modify `ToolEnabledAI`/`BaseAgent`/`Coordinator` to manage `conversation_id`s and pass the chosen `ConversationStore`.
      - **Detail:** Define how `conversation_id` (e.g., from Telegram `chat_id`) propagates through the call chain to `ConversationManager`.

## Phase 2: Tooling & Telegram Integration

- **Objective:** Implement specific tools (via MCP interfaces) and the Telegram UI.
- **Tasks:**
  1.  **Configure MCP Tools Interfaces:**
      - Define required tool interfaces (Email Sender, Web Search, RAG Memory, etc.) in `mcp.yml`.
      - Update `url` fields to point to the actual deployed MCP server endpoints (VM-hosted or potentially Lambda/API Gateway).
      - _Note: Actual MCP server implementation is out of scope for this plan._
  2.  **Implement Telegram Bot Integration:**
      - Create `telegram_bot.py` (runs on VM).
      - Add `python-telegram-bot` dependency.
      - Connect to Telegram API (configure bot token securely).
      - **Detail:** Define initialization logic - how are `Coordinator` and `ConversationStore` instances created and injected?
      - Handle incoming messages, extract `chat_id` (as `conversation_id`).
      - Call `Coordinator.process_request` correctly.
      - **Detail:** Implement robust error handling for Telegram API calls and Coordinator interactions.
      - Send response back via Telegram API.
      - **(Optional) Voice Handling:** Integrate STT/TTS capabilities.

## Phase 3: Deployment & Operations (VM Focus)

- **Objective:** Deploy the core framework reliably and cost-effectively on an always-on VM, ensuring responsiveness.
- **Tasks:**
  1.  **VM Provisioning:** Choose provider (Lightsail, EC2 Nano/Micro), provision small Linux instance.
      - **Attention:** Estimate initial resource needs (CPU/RAM).
  2.  **Dockerization (Recommended):**
      - Create `Dockerfile` for the core framework.
      - Create `docker-compose.yml` for the main app service (and local MCP servers if running on same VM).
  3.  **Configuration Management:**
      - Externalize secrets (API Keys, DB creds, Bot Tokens) via environment variables.
      - Update `config.yml`/`mcp.yml` for production environments.
  4.  **Security Hardening:**
      - Configure VM firewall rules (allow only necessary ports).
      - Set up SSH key-based access.
      - Assign minimal necessary IAM permissions (e.g., via IAM Role) if accessing AWS services like DynamoDB.
  5.  **Deployment Process:** Script `git pull`, `docker build`, `docker-compose up -d --build` (or similar).
  6.  **Process Management:** Use `docker-compose` with `restart: unless-stopped`.
  7.  **Logging:** Configure Docker logging driver (e.g., `awslogs` for CloudWatch). Ensure app logs write to stdout/stderr.
  8.  **(If using Lambdas for Tools):** Manage deployment of Lambda/API Gateway separately.

## Phase 4: Future Enhancements (Post-MVP)

- **Objective:** Improve performance, cost-efficiency, and capabilities based on usage.
- **Potential Tasks (Deferred):**
  - **Caching:** LLM response caching, RAG query caching.
  - **Advanced Model Selection:** Dynamic `UseCase` detection per-request.
  - **Advanced Coordinator:** Agent selection logic, parallel execution.
  - **Monitoring & Alerting:** CloudWatch alarms, application-level metrics.
  - **CI/CD:** Automate testing and deployment.
  - **Refined RAG Memory:** Improve conversational memory logic & retrieval.
  - **Optimize Tool Backends:** Analyze performance/cost of tool backend hosting.
