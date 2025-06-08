#!/bin/bash

# Comprehensive .zshrc Verification Script for Zed Terminal
# This script tests if .zshrc is properly loaded and all configurations are working

set -e

echo "🔍 Verifying .zshrc Configuration in Zed Terminal"
echo "=================================================="

# Test 1: Basic Shell Environment
echo "📋 Test 1: Basic Shell Environment"
echo "SHELL: $SHELL"
echo "ZSH: $ZSH"
echo "USER: $USER"
echo "HOME: $HOME"
echo "PWD: $PWD"

# Test 2: Oh My Zsh Configuration
echo ""
echo "📋 Test 2: Oh My Zsh Configuration"
if [ -d "$HOME/.oh-my-zsh" ]; then
    echo "✅ Oh My Zsh directory exists: $HOME/.oh-my-zsh"
    if [ -n "$ZSH_THEME" ]; then
        echo "✅ ZSH_THEME is set: $ZSH_THEME"
    else
        echo "⚠️  ZSH_THEME not set"
    fi
    if [ -n "$plugins" ]; then
        echo "✅ Plugins are configured"
    else
        echo "⚠️  No plugins detected"
    fi
else
    echo "❌ Oh My Zsh not found"
fi

# Test 3: Powerlevel10k Theme
echo ""
echo "📋 Test 3: Powerlevel10k Theme"
if [ -d "$HOME/.oh-my-zsh/custom/themes/powerlevel10k" ]; then
    echo "✅ Powerlevel10k theme directory exists"
    if [ -f "$HOME/.p10k.zsh" ]; then
        echo "✅ Powerlevel10k configuration file exists"
    else
        echo "⚠️  Powerlevel10k config not found at ~/.p10k.zsh"
    fi
else
    echo "⚠️  Powerlevel10k theme not installed"
fi

# Test 4: Essential Tools
echo ""
echo "📋 Test 4: Essential Tools Availability"
tools=("brew" "git" "python" "node" "npx" "fzf" "zoxide")
for tool in "${tools[@]}"; do
    if command -v "$tool" &> /dev/null; then
        version=$($tool --version 2>/dev/null | head -1 || echo "unknown")
        echo "✅ $tool: $version"
    else
        echo "❌ $tool: not found"
    fi
done

# Test 5: PATH Configuration
echo ""
echo "📋 Test 5: PATH Configuration"
echo "PATH components:"
echo "$PATH" | tr ':' '\n' | nl

# Test 6: Homebrew Integration
echo ""
echo "📋 Test 6: Homebrew Integration"
if command -v brew &> /dev/null; then
    brew_prefix=$(brew --prefix 2>/dev/null)
    echo "✅ Homebrew prefix: $brew_prefix"
    if [[ "$PATH" == *"$brew_prefix"* ]]; then
        echo "✅ Homebrew in PATH"
    else
        echo "⚠️  Homebrew not in PATH"
    fi
else
    echo "❌ Homebrew not available"
fi

# Test 7: FZF Integration
echo ""
echo "📋 Test 7: FZF Integration"
if command -v fzf &> /dev/null; then
    echo "✅ FZF command available"
    if [ -f "$(brew --prefix 2>/dev/null)/opt/fzf/shell/completion.zsh" ]; then
        echo "✅ FZF completion script exists"
    else
        echo "⚠️  FZF completion script not found"
    fi
    if [ -f "$(brew --prefix 2>/dev/null)/opt/fzf/shell/key-bindings.zsh" ]; then
        echo "✅ FZF key bindings script exists"
    else
        echo "⚠️  FZF key bindings script not found"
    fi
else
    echo "❌ FZF not available"
fi

# Test 8: Zoxide Integration
echo ""
echo "📋 Test 8: Zoxide Integration"
if command -v zoxide &> /dev/null; then
    echo "✅ Zoxide command available: $(zoxide --version)"
    if alias z &> /dev/null; then
        echo "✅ Zoxide 'z' alias configured"
    else
        echo "⚠️  Zoxide 'z' alias not found"
    fi
    if command -v zi &> /dev/null; then
        echo "✅ Zoxide 'zi' interactive command available"
    else
        echo "⚠️  Zoxide 'zi' interactive command not found"
    fi
else
    echo "❌ Zoxide not available"
fi

# Test 9: NVM Integration
echo ""
echo "📋 Test 9: NVM Integration"
if [ -d "$HOME/.nvm" ]; then
    echo "✅ NVM directory exists: $HOME/.nvm"
    if [ -s "$HOME/.nvm/nvm.sh" ]; then
        echo "✅ NVM script exists"
        if command -v nvm &> /dev/null; then
            echo "✅ NVM command available"
        else
            echo "⚠️  NVM command not loaded"
        fi
    else
        echo "⚠️  NVM script not found"
    fi
else
    echo "❌ NVM not installed"
fi

# Test 10: Python Configuration
echo ""
echo "📋 Test 10: Python Configuration"
python_paths=("/opt/homebrew/bin/python3.12" "$HOME/Library/Python/3.12/bin")
for py_path in "${python_paths[@]}"; do
    if [ -f "$py_path/python" ] || [ -f "$py_path" ]; then
        echo "✅ Python path exists: $py_path"
    else
        echo "⚠️  Python path not found: $py_path"
    fi
done

if [[ "$PATH" == *"$HOME/Library/Python/3.12/bin"* ]]; then
    echo "✅ Python user bin in PATH"
else
    echo "⚠️  Python user bin not in PATH"
fi

# Test 11: Google Cloud SDK
echo ""
echo "📋 Test 11: Google Cloud SDK"
gcloud_path="$HOME/dev/vertexai/google-cloud-sdk"
if [ -d "$gcloud_path" ]; then
    echo "✅ Google Cloud SDK directory exists"
    if [ -f "$gcloud_path/path.zsh.inc" ]; then
        echo "✅ Google Cloud SDK path script exists"
    else
        echo "⚠️  Google Cloud SDK path script not found"
    fi
    if [ -f "$gcloud_path/completion.zsh.inc" ]; then
        echo "✅ Google Cloud SDK completion script exists"
    else
        echo "⚠️  Google Cloud SDK completion script not found"
    fi
    if command -v gcloud &> /dev/null; then
        echo "✅ gcloud command available"
    else
        echo "⚠️  gcloud command not available"
    fi
else
    echo "⚠️  Google Cloud SDK directory not found"
fi

# Test 12: Docker Integration
echo ""
echo "📋 Test 12: Docker Integration"
if [ -d "$HOME/.docker/completions" ]; then
    echo "✅ Docker completions directory exists"
    if command -v docker &> /dev/null; then
        echo "✅ Docker command available"
    else
        echo "⚠️  Docker command not available"
    fi
else
    echo "⚠️  Docker completions not found"
fi

# Test 13: DevQ.ai Project Configuration
echo ""
echo "📋 Test 13: DevQ.ai Project Configuration"
devqai_vars=("DEVQAI_ROOT" "PYTHONPATH" "MCP_SERVERS_PATH" "PTOLEMIES_PATH")
for var in "${devqai_vars[@]}"; do
    if [ -n "${!var}" ]; then
        echo "✅ $var: ${!var}"
    else
        echo "⚠️  $var not set"
    fi
done

# Test 14: Database Configuration
echo ""
echo "📋 Test 14: Database Configuration"
db_vars=("SURREALDB_URL" "SURREALDB_USERNAME" "SURREALDB_PASSWORD" "DART_TOKEN")
for var in "${db_vars[@]}"; do
    if [ -n "${!var}" ]; then
        echo "✅ $var: [CONFIGURED]"
    else
        echo "⚠️  $var not set"
    fi
done

# Test 15: Project Aliases
echo ""
echo "📋 Test 15: DevQ.ai Project Aliases"
aliases=("ag" "bayes" "darwin" "nash" "ptolemies" "start-surreal" "start-dart" "devq-test")
for alias_name in "${aliases[@]}"; do
    if alias "$alias_name" &> /dev/null; then
        echo "✅ Alias '$alias_name' configured"
    else
        echo "⚠️  Alias '$alias_name' not found"
    fi
done

# Test 16: Custom Functions
echo ""
echo "📋 Test 16: Custom Functions"
functions=("show_env_vars" "new-component" "dart-test" "mcp-inspect")
for func in "${functions[@]}"; do
    if declare -f "$func" &> /dev/null; then
        echo "✅ Function '$func' available"
    else
        echo "⚠️  Function '$func' not found"
    fi
done

# Test 17: History Configuration
echo ""
echo "📋 Test 17: History Configuration"
echo "HISTSIZE: ${HISTSIZE:-not set}"
echo "SAVEHIST: ${SAVEHIST:-not set}"
echo "HISTFILE: ${HISTFILE:-not set}"

# Test 18: Completion System
echo ""
echo "📋 Test 18: Completion System"
if command -v compinit &> /dev/null; then
    echo "✅ Completion system (compinit) available"
else
    echo "⚠️  Completion system not loaded"
fi

# Test 19: File Existence Check
echo ""
echo "📋 Test 19: Configuration Files"
config_files=("$HOME/.zshrc" "$HOME/.p10k.zsh" "$(pwd)/.zshrc.devqai")
for file in "${config_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ Configuration file exists: $file"
    else
        echo "⚠️  Configuration file missing: $file"
    fi
done

# Test 20: Environment Summary
echo ""
echo "📋 Test 20: Environment Summary"
echo "Current directory: $(pwd)"
echo "Shell level: $SHLVL"
echo "Terminal: ${TERM_PROGRAM:-unknown}"
if [ -n "$ZED_TERMINAL" ]; then
    echo "✅ Running in Zed terminal"
else
    echo "⚠️  ZED_TERMINAL variable not set"
fi

# Final Report
echo ""
echo "🎯 Verification Complete!"
echo "========================="

# Count successful tests
echo "Run this script in your Zed terminal to verify .zshrc loading:"
echo "cd /Users/dionedge/devqai && ./verify-zshrc-in-zed.sh"
echo ""
echo "If you see mostly ✅ marks, your .zshrc is properly loaded!"
echo "If you see ⚠️  or ❌ marks, there may be configuration issues."
