#!/bin/bash

# Quick Zed Terminal Verification Test
# Run this with: ./quick-test.sh

echo "🔍 Quick Zed Terminal Test"
echo "========================="

# Test 1: Basic environment
echo "1. Environment: $([ -n "$DEVQAI_ROOT" ] && echo "✅ DEVQAI_ROOT set" || echo "❌ DEVQAI_ROOT missing")"

# Test 2: Zoxide
echo "2. Zoxide: $(command -v zoxide >/dev/null 2>&1 && echo "✅ Available" || echo "❌ Missing")"

# Test 3: Essential tools
echo "3. Tools: $(command -v brew git node npx >/dev/null 2>&1 && echo "✅ All found" || echo "❌ Some missing")"

# Test 4: DevQ.ai aliases
echo "4. Aliases: $(alias ag >/dev/null 2>&1 && echo "✅ DevQ.ai aliases loaded" || echo "❌ Aliases missing")"

# Test 5: Functions
echo "5. Functions: $(declare -f show_env_vars >/dev/null 2>&1 && echo "✅ Custom functions loaded" || echo "❌ Functions missing")"

# Test 6: Dart AI
echo "6. Dart AI: $([ -n "$DART_TOKEN" ] && echo "✅ Token configured" || echo "❌ Token missing")"

# Test 7: Oh My Zsh
echo "7. Oh My Zsh: $([ -n "$ZSH" ] && echo "✅ Loaded" || echo "❌ Not loaded")"

echo ""
echo "🎯 Quick Fix (if needed):"
echo "source ~/.zshrc && source .zshrc.devqai"
