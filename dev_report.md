## Claude Code Core Development Tools (Built-in)

- `task` - Launch specialized agents for complex searches and analysis
- `bash` - Execute shell commands with git, file operations, and system tasks
- `glob` - Fast file pattern matching (e.g., **/*.js, src/**/*.ts)
- `grep` - Content search using regex patterns across codebases
- `ls` - Directory listing and file system navigation
- `read` - File content reading including images and notebooks
- `edit/multiedit` - Precise file editing with find-and-replace
- `write` - Create new files and overwrite existing ones
- `todoread/todowrite` - Task management and progress tracking

### MCP (Model Context Protocol) Servers Available

**NPX-Based Core Servers**

- `filesystem` - File read/write operations
- `git` - Version control operations and commits
- `fetch` - API calls and external resource access
- `memory` - Persistent memory across sessions
- `sequentialthinking` - Enhanced step-by-step problem solving
- `github` - GitHub API integration for repositories and issues
- `inspector` - Debug MCP server connections

**Local Python-Based Servers**

- `bayes` - Bayesian inference and statistical modeling
- `ptolemies` - Custom knowledge base with SurrealDB integration
- `context7` - Advanced contextual reasoning with Redis memory
- `crawl4ai` - Intelligent web scraping and content extraction
- `dart` - Dart/Flutter development tools
- `agentql` - Web automation using natural language queries
- `calendar` - Google Calendar integration
- `jupyter` - Notebook execution and data science workflows
- `stripe` - Payment processing integration
- `shadcn`-ui - React component library integration
- `magic` - AI-powered code generation utilities

**Specialized Solvers**

- `solver-z3` - Z3 theorem prover for formal verification
- `solver-pysat` - Boolean satisfiability problem solver
- `solver-mzn` - MiniZinc constraint satisfaction solver

**Database & Infrastructure**

- `surrealdb` - Multi-model database operations with graph capabilities
- `memory` - Persistent memory management with contextual recall
- `registry` - Official MCP server registry and discovery

**Key Capabilities**

- Multi-modal - Handle text, code, images, and data
- Database Integration - SurrealDB, Redis, Neo4j support
- AI/ML Tools - Multiple LLM providers and specialized analysis
- Web & Automation - Scraping, browser automation, API integration
- Development - Full-stack development with testing and deployment tools

### Tool Access Configuration

**`.claude/settings.local.json`**

- Enables MCP server auto-discovery: "enableAllProjectMcpServers": true
- Sets permissions: Bash command allowlist for security
- Does NOT define which MCP servers to load - just enables discovery

**`CLAUDE.md`**

- Documentation only - describes what should be available
- Reference specification - not executable configuration
- Tool inventory - comprehensive list of intended capabilities

**`mcp/mcp-servers.json`**

- Actual MCP server definitions - the authoritative configuration
- Executable specifications - commands, arguments, environment variables
- Real server registry - what Claude Code actually loads

**How It Works Together**

1. Claude Code reads `mcp-servers.json` for actual MCP server definitions
2. `settings.local.json` enables the auto-discovery of those servers
3. `CLAUDE.md` documents what should be available (alignment reference)

**Built-in Tools vs MCP Tools**

- Built-in tools (`task`, `bash`, `glob`, etc.) are always available - no configuration needed
- MCP tools require the configuration chain above to be accessible

## Current Status

✅ All three files are aligned after our standardization work
✅ MCP servers are configured and ready to use
✅ Built-in tools work immediately regardless of configuration

The capabilities are accessible through this configuration chain, but `mcp-servers.json` is the actual enabler while the others provide discovery and documentation.

### Zed Editor Configuration (`.zed/settings.json`)

- IDE-specific: Configures Zed editor behavior
- Development environment: Terminal, formatting, LSP, themes
- Local MCP integration: Direct MCP server configuration for Zed's AI features
- Project environment: Sets up PYTHONPATH, working directories, shell initialization

### Claude Code Configuration

- AI assistant-specific: Configures Claude Code's tool access
- MCP server registry: Centralized server definitions
- Permission system: Security controls for tool usage

### Environment Variables

- Zed: Sets basic env vars in terminal
- Claude: Uses comprehensive `.env` files with 94+ variables
- Overlap: Both reference MCP paths and database configs

### MCP Server Definitions

- Zed: Defines `mcpServers` for editor AI features
- Claude: Uses `mcp-servers.json` for 18+ comprehensive servers
- Different: Zed has 1 server (dart), Claude has 18+ servers

### File Structure Integration

devqai/
├── `.zed/settings.json`           # Zed editor configuration
├── `.claude/settings.local.json`  # Claude Code permissions
├── `CLAUDE.md`                    # Claude Code documentation
├── `mcp/mcp-servers.json`         # Claude Code MCP servers
└── `.env`                         # Master environment (both can use)

### Relationship Summary

- Independent systems with different purposes
- Zed: Editor/IDE configuration for development workflow
- Claude: AI assistant tool access and capabilities
- Shared: Environment variables and project paths
- Complementary: Both can leverage the unified `.env` configuration

The Zed configuration enhances the development environment, while the Claude configuration enables AI-powered assistance tools. The Zed terminal configuration works perfectly when properly sourced, providing enhanced development commands and proper environment variables for the DevQ.ai ecosystem.  Zed is running with multiple MCP servers already loaded.

```
> source .zshrc.devqai
🚀 DevQ.ai development environment loaded!
📁 Project root: /Users/dionedge/devqai
🔧 Available commands:
   • Quick nav: ag, bayes, darwin, nash, ptolemies, breiman, gompertz, tokenator
   • Zoxide: z <dir>, zi (interactive), zz (back), zq (query), zr (remove)
   • MCP: start-context7, start-crawl4ai, start-ptolemies, start-dart, mcp-inspect
   • Dev tools: devq-test, devq-format, devq-lint, start-surreal, verify-db
   • Utils: new-component, find-dir, find-edit, show_env_vars, dart-test
```
Zed terminal now seamlessly emulates the existing `.zshrc` setup while adding zoxide smart navigation and comprehensive DevQ.ai development tools. Everything loads automatically and works exactly as intended.
