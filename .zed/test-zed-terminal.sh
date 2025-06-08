#!/bin/bash

# Zed Terminal Test Script
# Run this script in Zed terminal to verify .zshrc loading

echo "🔍 Testing Zed Terminal Configuration"
echo "======================================"

# Test shell environment
echo ""
echo "📋 Shell Environment:"
echo "SHELL: $SHELL"
echo "PWD: $PWD"
echo "USER: $USER"

# Test if we're in zsh
if [ -n "$ZSH_VERSION" ]; then
    echo "✅ Running in zsh: $ZSH_VERSION"
else
    echo "❌ Not running in zsh"
fi

# Test Oh My Zsh
echo ""
echo "📋 Oh My Zsh:"
if [ -n "$ZSH" ] && [ -d "$ZSH" ]; then
    echo "✅ Oh My Zsh loaded: $ZSH"
    echo "Theme: ${ZSH_THEME:-not set}"
else
    echo "❌ Oh My Zsh not loaded"
fi

# Test essential tools
echo ""
echo "📋 Essential Tools:"
for tool in brew git python node npx zoxide fzf; do
    if command -v $tool >/dev/null 2>&1; then
        echo "✅ $tool: $(command -v $tool)"
    else
        echo "❌ $tool: not found"
    fi
done

# Test DevQ.ai environment
echo ""
echo "📋 DevQ.ai Environment:"
echo "DEVQAI_ROOT: ${DEVQAI_ROOT:-not set}"
echo "PYTHONPATH: ${PYTHONPATH:-not set}"
echo "MCP_SERVERS_PATH: ${MCP_SERVERS_PATH:-not set}"
echo "PTOLEMIES_PATH: ${PTOLEMIES_PATH:-not set}"
echo "DART_TOKEN: ${DART_TOKEN:+[CONFIGURED]}"

# Test aliases
echo ""
echo "📋 DevQ.ai Aliases:"
for alias_name in ag bayes darwin nash ptolemies start-surreal start-dart; do
    if alias $alias_name >/dev/null 2>&1; then
        echo "✅ $alias_name: $(alias $alias_name | cut -d'=' -f2-)"
    else
        echo "❌ $alias_name: not found"
    fi
done

# Test zoxide
echo ""
echo "📋 Zoxide Integration:"
if command -v zoxide >/dev/null 2>&1; then
    echo "✅ Zoxide installed: $(zoxide --version)"
    if command -v z >/dev/null 2>&1; then
        echo "✅ z command available"
    else
        echo "❌ z command not found"
    fi
else
    echo "❌ Zoxide not installed"
fi

# Test functions
echo ""
echo "📋 Custom Functions:"
for func in show_env_vars new-component dart-test mcp-inspect; do
    if declare -f $func >/dev/null 2>&1; then
        echo "✅ $func: available"
    else
        echo "❌ $func: not found"
    fi
done

# Test PATH
echo ""
echo "📋 PATH Configuration:"
if echo "$PATH" | grep -q "/opt/homebrew/bin"; then
    echo "✅ Homebrew in PATH"
else
    echo "❌ Homebrew not in PATH"
fi

if echo "$PATH" | grep -q "$HOME/.local/bin"; then
    echo "✅ Local bin in PATH"
else
    echo "❌ Local bin not in PATH"
fi

# Final summary
echo ""
echo "🎯 Test Summary"
echo "==============="
echo "If you see mostly ✅ marks above, your Zed terminal is properly configured!"
echo "If you see ❌ marks, check your .zshrc and .zshrc.devqai files."
echo ""
echo "To manually source configuration:"
echo "source ~/.zshrc && source .zshrc.devqai"
