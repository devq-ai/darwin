# Zed Terminal Configuration for DevQ.ai
# Simplified version compatible with Zed's terminal environment

# Basic environment setup
export ZSH="$HOME/.oh-my-zsh"
export EDITOR="code"

# DevQ.ai Project Configuration
if [[ -d "$(pwd)/devqai" ]]; then
    export DEVQAI_ROOT="$(pwd)/devqai"
elif [[ "$(pwd)" == *"devqai"* ]]; then
    export DEVQAI_ROOT="$(pwd)"
fi

if [[ -n "$DEVQAI_ROOT" ]]; then
    export PYTHONPATH="$DEVQAI_ROOT:$PYTHONPATH"
    export MCP_SERVERS_PATH="$DEVQAI_ROOT/mcp/mcp-servers"
    export PTOLEMIES_PATH="$DEVQAI_ROOT/ptolemies"
fi

# Database Configuration
export SURREALDB_URL="ws://localhost:8000/rpc"
export SURREALDB_USERNAME="root"
export SURREALDB_PASSWORD="root"
export SURREALDB_NAMESPACE="ptolemies"
export SURREALDB_DATABASE="knowledge"

# History settings
HISTSIZE=10000
SAVEHIST=10000
HISTFILE=~/.zsh_history
setopt SHARE_HISTORY
setopt HIST_IGNORE_DUPS
setopt AUTO_CD
setopt CORRECT

# Load Oh My Zsh if available
if [[ -d "$ZSH" ]]; then
    ZSH_THEME="robbyrussell"  # Fallback theme for Zed
    plugins=(git z zsh-autosuggestions zsh-syntax-highlighting)
    source $ZSH/oh-my-zsh.sh
fi

# PATH setup
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PATH="$HOME/Library/Python/3.12/bin:$PATH"

# Remove duplicates from PATH
PATH=$(echo "$PATH" | awk -v RS=':' -v ORS=":" '!a[$1]++' | sed 's/:$//')

# Homebrew setup
if command -v brew &> /dev/null; then
    eval "$(brew shellenv)"
fi

# Initialize zoxide if available
if command -v zoxide &> /dev/null; then
    eval "$(zoxide init zsh)"
fi

# NVM setup
export NVM_DIR="$HOME/.nvm"
[[ -s "$NVM_DIR/nvm.sh" ]] && source "$NVM_DIR/nvm.sh"
[[ -s "$NVM_DIR/bash_completion" ]] && source "$NVM_DIR/bash_completion"

# fzf setup
if [[ -f "$(brew --prefix 2>/dev/null)/opt/fzf/shell/completion.zsh" ]]; then
    source "$(brew --prefix)/opt/fzf/shell/completion.zsh"
    source "$(brew --prefix)/opt/fzf/shell/key-bindings.zsh"
fi

# Docker completions
if [[ -d "$HOME/.docker/completions" ]]; then
    fpath=($HOME/.docker/completions $fpath)
fi

# Load completions
autoload -Uz compinit && compinit

# DevQ.ai Project Aliases
if [[ -n "$DEVQAI_ROOT" ]]; then
    alias devq="cd $DEVQAI_ROOT"
    alias devq-test="python -m pytest"
    alias devq-format="black . && isort ."
    alias devq-lint="flake8 . && mypy ."

    # Quick navigation with zoxide or cd
    if command -v zoxide &> /dev/null; then
        alias ag="z agentical"
        alias bayes="z bayes"
        alias darwin="z darwin"
        alias nash="z nash"
        alias ptolemies="z ptolemies"
        alias breiman="z breiman"
        alias gompertz="z gompertz"
        alias tokenator="z tokenator"
        alias zz="z -"
        alias zi="zi"
    else
        alias ag="cd $DEVQAI_ROOT/agentical"
        alias bayes="cd $DEVQAI_ROOT/bayes"
        alias darwin="cd $DEVQAI_ROOT/darwin"
        alias nash="cd $DEVQAI_ROOT/nash"
        alias ptolemies="cd $DEVQAI_ROOT/ptolemies"
        alias breiman="cd $DEVQAI_ROOT/breiman"
        alias gompertz="cd $DEVQAI_ROOT/gompertz"
        alias tokenator="cd $DEVQAI_ROOT/tokenator"
    fi

    # MCP Server Management
    alias start-context7="cd $MCP_SERVERS_PATH/context7-mcp && python -m context7_mcp.server"
    alias start-crawl4ai="cd $MCP_SERVERS_PATH/crawl4ai-mcp && python -m crawl4ai_mcp.server"
    alias start-ptolemies="cd $PTOLEMIES_PATH && python -m ptolemies.mcp.ptolemies_mcp"

    # Database helpers
    alias start-surreal="surreal start --log trace --user root --pass root memory"
    alias verify-db="cd $PTOLEMIES_PATH && python verify-database.py"
    alias setup-db="cd $PTOLEMIES_PATH && ./setup-database.sh"
fi

# Git shortcuts
alias gs="git status"
alias gp="git pull"
alias gc="git commit -m"
alias gd="git diff"
alias glog="git log --oneline --graph --decorate"

# General aliases
alias ll="ls -la"
alias la="ls -A"
alias l="ls -CF"
alias ..="cd .."
alias ...="cd ../.."

# Python helpers
alias activate-venv="source venv/bin/activate"

# Function to show environment variables
function show_env_vars() {
    echo "USER: $USER"
    echo "HOME: $HOME"
    echo "SHELL: $SHELL"
    echo "PWD: $PWD"
    echo "EDITOR: $EDITOR"
    echo "VIRTUAL_ENV: $VIRTUAL_ENV"
    [[ -n "$DEVQAI_ROOT" ]] && echo "DEVQAI_ROOT: $DEVQAI_ROOT"
}

# Function to create new DevQ.ai components
function new-component() {
    if [[ -z "$1" ]] || [[ -z "$DEVQAI_ROOT" ]]; then
        echo "Usage: new-component <component-name>"
        echo "Must be run from DevQ.ai project directory"
        return 1
    fi
    mkdir -p "$DEVQAI_ROOT/$1"
    cd "$DEVQAI_ROOT/$1"
    echo "Created and navigated to $1"
}

# Load project-specific .env if it exists
if [[ -n "$DEVQAI_ROOT" ]] && [[ -f "$DEVQAI_ROOT/.env" ]]; then
    set -a
    source "$DEVQAI_ROOT/.env"
    set +a
fi

# Welcome message
if [[ -n "$DEVQAI_ROOT" ]]; then
    echo "🚀 DevQ.ai terminal ready!"
    echo "📁 Project: $DEVQAI_ROOT"
    echo "💡 Try: ag, bayes, darwin, nash, ptolemies, start-surreal"
fi
