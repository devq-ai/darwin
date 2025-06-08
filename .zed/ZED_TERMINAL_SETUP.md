# Zed Terminal Configuration for DevQ.ai

This guide will help you configure Zed's terminal to use your existing zsh configuration with zoxide support.

## Method 1: Direct Zed Terminal Configuration

### Step 1: Configure Zed Settings

Open Zed settings (`Cmd+,`) and add the following terminal configuration:

```json
{
  "terminal": {
    "shell": {
      "program": "/bin/zsh",
      "args": ["-l"]
    },
    "working_directory": "current_project_directory",
    "env": {
      "DEVQAI_ROOT": "$(pwd)"
    }
  }
}
```

### Step 2: Create Zed-Specific ZSH Configuration

Add this to your `~/.zshrc` (or create a separate `~/.zshrc.zed`):

```bash
# Zed Terminal Detection and Configuration
if [[ "$TERM_PROGRAM" == "zed" ]] || [[ -n "$ZED_TERMINAL" ]]; then
    # DevQ.ai Project Detection
    if [[ "$(pwd)" == *"devqai"* ]]; then
        export DEVQAI_ROOT="$(pwd)"
        export PYTHONPATH="$DEVQAI_ROOT:$PYTHONPATH"
        export MCP_SERVERS_PATH="$DEVQAI_ROOT/mcp/mcp-servers"
        export PTOLEMIES_PATH="$DEVQAI_ROOT/ptolemies"

        # Load project-specific configuration
        [[ -f "$DEVQAI_ROOT/.zshrc.devqai" ]] && source "$DEVQAI_ROOT/.zshrc.devqai"
    fi
fi
```

## Method 2: Project-Specific Zed Configuration

### Create .zed/settings.json in your devqai directory:

```json
{
  "terminal": {
    "shell": {
      "program": "/bin/zsh",
      "args": ["-l", "-c", "source ~/.zshrc && source .zshrc.devqai 2>/dev/null || true; exec zsh"]
    },
    "working_directory": "current_project_directory"
  }
}
```

## Method 3: Simple Startup Script

Create `devqai/start-zed-terminal.sh`:

```bash
#!/bin/bash
cd "$(dirname "$0")"
exec /bin/zsh -l -c "source ~/.zshrc; source .zshrc.devqai 2>/dev/null || true; exec zsh"
```

Then configure Zed to use this script:

```json
{
  "terminal": {
    "shell": {
      "program": "/Users/dionedge/devqai/start-zed-terminal.sh"
    }
  }
}
```

## Verification

Once configured, your Zed terminal should have:

1. **Powerlevel10k theme** (if configured in your main .zshrc)
2. **zoxide support** with commands:
   - `z <directory>` - Smart navigation
   - `zi` - Interactive directory picker
   - `zz` - Go back to previous directory
3. **DevQ.ai aliases**:
   - `ag`, `bayes`, `darwin`, `nash`, `ptolemies` - Quick navigation
   - `start-surreal`, `verify-db`, `setup-db` - Database commands
   - `start-context7`, `start-crawl4ai`, `start-ptolemies` - MCP servers

## Troubleshooting

### Terminal doesn't load configuration
- Ensure `/bin/zsh` is your default shell: `chsh -s /bin/zsh`
- Check that oh-my-zsh is installed: `ls ~/.oh-my-zsh`

### zoxide not working
- Install zoxide: `brew install zoxide`
- Add to .zshrc: `eval "$(zoxide init zsh)"`

### Powerlevel10k not showing
- Install theme: `git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ~/.oh-my-zsh/custom/themes/powerlevel10k`
- Set in .zshrc: `ZSH_THEME="powerlevel10k/powerlevel10k"`

### Project aliases not loading
- Ensure you're in the devqai directory when opening Zed
- Check that `.zshrc.devqai` exists in the project root

## Testing Your Setup

Run these commands in your Zed terminal:

```bash
# Test zoxide
z ptolemies  # Should navigate to ptolemies directory

# Test project aliases
start-surreal  # Should start SurrealDB

# Test environment
show_env_vars  # Should show DEVQAI_ROOT and other variables
```

## Environment Variables Set

When properly configured, these variables will be available:

- `DEVQAI_ROOT` - Project root directory
- `PYTHONPATH` - Includes project in Python path
- `MCP_SERVERS_PATH` - Path to MCP servers
- `PTOLEMIES_PATH` - Path to Ptolemies knowledge base
- `SURREALDB_*` - Database connection settings
