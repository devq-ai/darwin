# Zoxide Usage Guide for DevQ.ai

Zoxide is a smarter `cd` command that learns your habits and helps you navigate directories faster.

## Installation & Setup

Zoxide is already installed via Homebrew. To initialize it in your shell:

```bash
# Add to your .zshrc
eval "$(zoxide init zsh)"
```

## Basic Commands

### `z` - Smart Navigation
```bash
z ptolemies      # Jump to ptolemies directory
z mcp           # Jump to mcp directory
z context       # Jump to context7-mcp (partial match)
z bayes         # Jump to bayes directory
```

### `zi` - Interactive Selection
```bash
zi              # Opens interactive directory picker with fzf
```

### `zoxide` Direct Commands
```bash
zoxide query ptolemies    # Show matching directories
zoxide add /path/to/dir   # Manually add directory
zoxide remove /path       # Remove directory from database
zoxide query --list       # List all tracked directories
zoxide query --stats      # Show access statistics
```

## DevQ.ai Project Aliases

These aliases are available when using the project configuration:

```bash
# Quick navigation (using zoxide)
ag              # Jump to agentical
bayes           # Jump to bayes
darwin          # Jump to darwin
nash            # Jump to nash
ptolemies       # Jump to ptolemies
breiman         # Jump to breiman
gompertz        # Jump to gompertz
tokenator       # Jump to tokenator

# Zoxide shortcuts
zz              # Go back to previous directory (z -)
zi              # Interactive directory picker
zq              # Query zoxide database (zoxide query)
zr              # Remove directory (zoxide remove)
```

## Advanced Usage

### Scoring System
Zoxide uses a scoring algorithm based on:
- **Frequency** - How often you visit a directory
- **Recency** - How recently you visited it
- Directories you visit more often rank higher

### Partial Matching
```bash
z ptol          # Matches ptolemies
z ctx           # Matches context7-mcp
z mcps          # Matches mcp-servers
```

### Multiple Matches
When multiple directories match, zoxide picks the highest-ranked one. Use `zi` to see all options.

### Environment Variables
```bash
export _ZO_DATA_DIR="$HOME/.local/share/zoxide"    # Database location
export _ZO_ECHO=1                                  # Print matched directory
export _ZO_EXCLUDE_DIRS="$HOME/.git:$HOME/node_modules"  # Exclude patterns
```

## Integration with DevQ.ai Workflow

### MCP Server Navigation
```bash
z context7      # Jump to context7-mcp
z crawl4ai      # Jump to crawl4ai-mcp
z ptolemies     # Jump to ptolemies for MCP server
```

### Quick Project Setup
```bash
z devqai        # Jump to project root
z ptolemies     # Jump to knowledge base
start-surreal   # Start database (uses alias)
verify-db       # Verify database setup
```

### Development Workflow
```bash
z bayes         # Work on Bayesian components
devq-test       # Run tests (project alias)
devq-format     # Format code (project alias)
z nash          # Switch to Nash equilibrium work
```

## Tips & Tricks

1. **Build History**: Use `cd` normally for the first few visits, then switch to `z`
2. **Case Insensitive**: `z PTOLEMIES` works the same as `z ptolemies`
3. **Substring Matching**: `z ptol` matches `ptolemies`
4. **Tab Completion**: Use tab completion with `z` for available matches
5. **Backup Navigation**: If `z` fails, it falls back to `cd` behavior

## Troubleshooting

### Database Issues
```bash
# Rebuild database
zoxide import --from autojump ~/.local/share/autojump/autojump.txt
# Or start fresh
rm -rf ~/.local/share/zoxide/db.zo
```

### Not Finding Directories
```bash
# Manually add frequently used directories
zoxide add $DEVQAI_ROOT/ptolemies
zoxide add $DEVQAI_ROOT/mcp/mcp-servers
```

### Conflicts with Other Tools
```bash
# Disable conflicting plugins in .zshrc
# Remove 'autojump' from plugins if using both
plugins=(git z zsh-autosuggestions zsh-syntax-highlighting zsh-completions)
```

## Performance

- Zoxide is written in Rust and is extremely fast
- Database queries are typically under 1ms
- Works with directories containing thousands of subdirectories
- Minimal memory footprint

## Comparison with Alternatives

| Tool | Speed | Learning | Fuzzy Search | Active Development |
|------|-------|----------|--------------|-------------------|
| zoxide | ⚡ Fast | ✅ Yes | ✅ Yes | ✅ Active |
| autojump | 🐌 Slow | ✅ Yes | ❌ No | ⚠️ Maintenance |
| z.sh | 🚀 Fast | ✅ Yes | ❌ No | ⚠️ Maintenance |
| fasd | 🚀 Fast | ✅ Yes | ❌ No | ❌ Deprecated |

Zoxide is the modern successor to these tools, offering the best performance and features.
