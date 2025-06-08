# Zed Terminal .zshrc Verification Guide

## Current Configuration Status

Your Zed terminal is configured in `.zed/settings.json` with the following shell setup:

```json
{
  "terminal": {
    "shell": {
      "program": "/bin/zsh",
      "args": ["-l", "-c", "export DEVQAI_ROOT=/Users/dionedge/devqai && cd $DEVQAI_ROOT && source ~/.zshrc && source .zshrc.devqai 2>/dev/null || true && zsh -i"]
    },
    "working_directory": "current_project_directory",
    "env": {
      "DEVQAI_ROOT": "/Users/dionedge/devqai",
      "PYTHONPATH": "/Users/dionedge/devqai:$PYTHONPATH",
      "MCP_SERVERS_PATH": "/Users/dionedge/devqai/mcp/mcp-servers",
      "PTOLEMIES_PATH": "/Users/dionedge/devqai/ptolemies",
      "SURREALDB_URL": "ws://localhost:8000/rpc",
      "SURREALDB_USERNAME": "root",
      "SURREALDB_PASSWORD": "root",
      "SURREALDB_NAMESPACE": "ptolemies",
      "SURREALDB_DATABASE": "knowledge",
      "DART_TOKEN": "dsa_1a21dba13961ac8abbe58ea7f9cb7d5621148dc2f3c79a9d346ef40430795e8f"
    }
  }
}
```

## Verification Tests

### Test 1: Manual Verification in Zed Terminal

Open a new terminal in Zed and run:

```bash
# Check if we're in the right directory
pwd

# Check if environment variables are set
echo "DEVQAI_ROOT: $DEVQAI_ROOT"
echo "DART_TOKEN: ${DART_TOKEN:0:10}..."

# Check if .zshrc was loaded
echo "ZSH: $ZSH"
echo "Shell: $SHELL"

# Test if zoxide is available
which zoxide
zoxide --version

# Test DevQ.ai aliases
alias ag
alias start-dart

# Test functions
type show_env_vars
type dart-test
```

### Test 2: Run the Test Script

```bash
./test-zed-terminal.sh
```

### Test 3: Check Configuration Loading

```bash
# Source configurations manually if needed
source ~/.zshrc
source .zshrc.devqai

# Verify welcome message appears
# Should see: "🚀 DevQ.ai development environment loaded!"
```

## Expected Results

When properly configured, you should see:

✅ **Environment Variables Set:**
- `DEVQAI_ROOT=/Users/dionedge/devqai`
- `DART_TOKEN=[CONFIGURED]`
- `MCP_SERVERS_PATH=/Users/dionedge/devqai/mcp/mcp-servers`
- `PTOLEMIES_PATH=/Users/dionedge/devqai/ptolemies`

✅ **Tools Available:**
- `zoxide` command for smart navigation
- `brew`, `git`, `node`, `npx` in PATH
- DevQ.ai welcome message on terminal start

✅ **Aliases Working:**
- `ag` → Navigate to agentical (via zoxide or cd)
- `bayes` → Navigate to bayes
- `start-dart` → Start Dart AI MCP server
- `start-surreal` → Start SurrealDB
- `devq-test` → Run project tests

✅ **Functions Available:**
- `show_env_vars` → Display environment information
- `dart-test` → Test Dart AI configuration
- `new-component` → Create new project component
- `mcp-inspect` → Run MCP inspector

## Troubleshooting

### Issue: Aliases Not Working

**Symptoms:** Commands like `ag`, `bayes`, `start-dart` not found

**Solution:**
```bash
# Manually source the configurations
source ~/.zshrc
source .zshrc.devqai

# Check if aliases are now available
alias ag
```

### Issue: Zoxide Not Working

**Symptoms:** `z` command not found, navigation aliases fail

**Check:**
```bash
# Verify zoxide is installed
which zoxide
zoxide --version

# Initialize zoxide manually
eval "$(zoxide init zsh)"

# Test navigation
z ptolemies
```

### Issue: Environment Variables Missing

**Symptoms:** `DEVQAI_ROOT` or `DART_TOKEN` not set

**Solution:**
```bash
# Set manually
export DEVQAI_ROOT=/Users/dionedge/devqai
export DART_TOKEN="dsa_1a21dba13961ac8abbe58ea7f9cb7d5621148dc2f3c79a9d346ef40430795e8f"

# Source project config
source .zshrc.devqai
```

### Issue: Oh My Zsh Not Loading

**Symptoms:** Plain prompt, no theme, missing completions

**Check:**
```bash
# Verify Oh My Zsh installation
ls ~/.oh-my-zsh

# Check if ZSH variable is set
echo $ZSH

# Source .zshrc manually
source ~/.zshrc
```

## Manual Setup Commands

If automatic loading fails, run these commands in your Zed terminal:

```bash
# 1. Navigate to project
cd /Users/dionedge/devqai

# 2. Set essential environment variables
export DEVQAI_ROOT=/Users/dionedge/devqai
export PYTHONPATH="$DEVQAI_ROOT:$PYTHONPATH"
export MCP_SERVERS_PATH="$DEVQAI_ROOT/mcp/mcp-servers"
export PTOLEMIES_PATH="$DEVQAI_ROOT/ptolemies"
export DART_TOKEN="dsa_1a21dba13961ac8abbe58ea7f9cb7d5621148dc2f3c79a9d346ef40430795e8f"

# 3. Source configurations
source ~/.zshrc
source .zshrc.devqai

# 4. Initialize zoxide
eval "$(zoxide init zsh)"

# 5. Verify setup
echo "🚀 Manual setup complete!"
alias ag
which zoxide
echo $DEVQAI_ROOT
```

## Alternative Configuration Methods

### Method 1: Global .zshrc Integration

Add to your `~/.zshrc`:

```bash
# DevQ.ai Auto-loader
if [[ "$PWD" == *"devqai"* ]] && [[ -f ".zshrc.devqai" ]]; then
    source ".zshrc.devqai"
fi
```

### Method 2: Zed Startup Script

Update `.zed/settings.json` to use the startup script:

```json
{
  "terminal": {
    "shell": {
      "program": "/Users/dionedge/devqai/zed-terminal.sh"
    }
  }
}
```

### Method 3: ZDOTDIR Override

Create a project-specific zsh configuration:

```bash
# In .zed/settings.json
{
  "terminal": {
    "env": {
      "ZDOTDIR": "/Users/dionedge/devqai/.zsh"
    }
  }
}
```

## Testing Commands Reference

### Quick Health Check
```bash
# One-liner to test everything
echo "Shell: $SHELL | DevQ: $DEVQAI_ROOT | Zoxide: $(which zoxide) | Aliases: $(alias ag 2>/dev/null && echo OK || echo FAIL)"
```

### Comprehensive Test
```bash
# Run full verification
./test-zed-terminal.sh > zed-terminal-test.log 2>&1
cat zed-terminal-test.log
```

### MCP Server Test
```bash
# Test Dart AI MCP server
dart-test

# Test other MCP servers
echo $MCP_SERVERS_PATH
ls $MCP_SERVERS_PATH
```

## Success Indicators

When everything is working correctly, opening a new Zed terminal should show:

```
🚀 DevQ.ai development environment loaded!
📁 Project root: /Users/dionedge/devqai
🔧 Available commands:
   • Quick nav: ag, bayes, darwin, nash, ptolemies, breiman, gompertz, tokenator
   • Zoxide: z <dir>, zi (interactive), zz (back), zq (query), zr (remove)
   • MCP: start-context7, start-crawl4ai, start-ptolemies, start-dart, mcp-inspect
   • Dev tools: devq-test, devq-format, devq-lint, start-surreal, verify-db
   • Utils: new-component, find-dir, find-edit, show_env_vars, dart-test
```

And all the listed commands should work immediately without additional setup.
