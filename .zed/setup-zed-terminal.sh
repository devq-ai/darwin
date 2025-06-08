#!/bin/bash

# DevQ.ai Zed Terminal Setup Script
# This script configures your Zed terminal to use the enhanced .zshrc configuration

set -e

DEVQAI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZSHRC_DEVQAI="$DEVQAI_ROOT/.zshrc.devqai"
HOME_ZSHRC="$HOME/.zshrc"

echo "🚀 Setting up Zed terminal for DevQ.ai..."

# Check if oh-my-zsh is installed
if [ ! -d "$HOME/.oh-my-zsh" ]; then
    echo "❌ Oh My Zsh is not installed. Please install it first:"
    echo "   sh -c \"\$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)\""
    exit 1
fi

# Check if Powerlevel10k is installed
if [ ! -d "$HOME/.oh-my-zsh/custom/themes/powerlevel10k" ]; then
    echo "📦 Installing Powerlevel10k theme..."
    git clone --depth=1 https://github.com/romkatv/powerlevel10k.git $HOME/.oh-my-zsh/custom/themes/powerlevel10k
else
    echo "✅ Powerlevel10k theme already installed"
fi

# Check if zsh-autosuggestions is installed
if [ ! -d "$HOME/.oh-my-zsh/custom/plugins/zsh-autosuggestions" ]; then
    echo "📦 Installing zsh-autosuggestions plugin..."
    git clone https://github.com/zsh-users/zsh-autosuggestions $HOME/.oh-my-zsh/custom/plugins/zsh-autosuggestions
else
    echo "✅ zsh-autosuggestions plugin already installed"
fi

# Check if zsh-syntax-highlighting is installed
if [ ! -d "$HOME/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting" ]; then
    echo "📦 Installing zsh-syntax-highlighting plugin..."
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git $HOME/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting
else
    echo "✅ zsh-syntax-highlighting plugin already installed"
fi

# Check if zsh-completions is installed
if [ ! -d "$HOME/.oh-my-zsh/custom/plugins/zsh-completions" ]; then
    echo "📦 Installing zsh-completions plugin..."
    git clone https://github.com/zsh-users/zsh-completions $HOME/.oh-my-zsh/custom/plugins/zsh-completions
else
    echo "✅ zsh-completions plugin already installed"
fi

# Check if zoxide is installed
if ! command -v zoxide &> /dev/null; then
    echo "📦 Installing zoxide..."
    brew install zoxide
else
    echo "✅ zoxide already installed"
fi

# Check if autojump is installed
if ! command -v autojump &> /dev/null; then
    echo "📦 Installing autojump..."
    brew install autojump
else
    echo "✅ autojump already installed"
fi

# Check if fzf is installed
if ! command -v fzf &> /dev/null; then
    echo "📦 Installing fzf..."
    brew install fzf
    $(brew --prefix)/opt/fzf/install
else
    echo "✅ fzf already installed"
fi

# Backup existing .zshrc if it exists and isn't already backed up
if [ -f "$HOME_ZSHRC" ] && [ ! -f "$HOME_ZSHRC.backup.devqai" ]; then
    echo "💾 Backing up existing .zshrc to .zshrc.backup.devqai"
    cp "$HOME_ZSHRC" "$HOME_ZSHRC.backup.devqai"
fi

# Create .zshrc.zed for Zed-specific configuration
ZSHRC_ZED="$HOME/.zshrc.zed"
echo "📝 Creating Zed-specific .zshrc configuration..."

cat > "$ZSHRC_ZED" << 'EOF'
# Zed Terminal Configuration for DevQ.ai
# This file is sourced when running zsh in Zed

# Enable Powerlevel10k instant prompt. Should stay close to the top of ~/.zshrc.
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

# Path to your oh-my-zsh installation.
export ZSH="$HOME/.oh-my-zsh"

# Set the Zsh theme to Powerlevel10k
ZSH_THEME="powerlevel10k/powerlevel10k"

# Enable automatic updates for oh-my-zsh without asking
zstyle ':omz:update' mode auto
zstyle ':omz:update' frequency 13

# Disable colors in `ls`
DISABLE_LS_COLORS="false"

# Set history timestamps format
HIST_STAMPS="yyyy/mm/dd"

# Load plugins including zoxide support
plugins=(git z zsh-autosuggestions zsh-syntax-highlighting zsh-completions autojump)

# Load oh-my-zsh
source $ZSH/oh-my-zsh.sh

# Set the manual path for man pages
export MANPATH="/usr/local/man:$MANPATH"

# Set preferred editor
if [[ -n $SSH_CONNECTION ]]; then
   export EDITOR='vim'
else
   export EDITOR='code'  # Use VS Code/Zed as default editor
fi

# fzf configuration
[[ -s $(brew --prefix)/opt/fzf/shell/keybindings.zsh ]] && source $(brew --prefix)/opt/fzf/shell/keybindings.zsh
[[ -s $(brew --prefix)/opt/fzf/shell/completion.zsh ]] && source $(brew --prefix)/opt/fzf/shell/completion.zsh

# Initialize zoxide (smart cd replacement)
if command -v zoxide &> /dev/null; then
    eval "$(zoxide init zsh)"
fi

# To customize prompt, run `p10k configure` or edit ~/.p10k.zsh.
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh

# Comprehensive PATH setup
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

# Remove duplicate PATH entries while preserving order
PATH=$(echo "$PATH" | awk -v RS=':' -v ORS=":" '!a[$1]++' | sed 's/:$//')

# Ensure Homebrew is properly set up
if command -v brew &> /dev/null; then
    eval "$(brew shellenv)"
fi

# Console Ninja
if [ -d ~/.console-ninja/.bin ]; then
    PATH=~/.console-ninja/.bin:$PATH
fi

# NVM setup
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

# Google Cloud SDK
if [ -f "$HOME/dev/vertexai/google-cloud-sdk/path.zsh.inc" ]; then
    source "$HOME/dev/vertexai/google-cloud-sdk/path.zsh.inc"
fi
if [ -f "$HOME/dev/vertexai/google-cloud-sdk/completion.zsh.inc" ]; then
    source "$HOME/dev/vertexai/google-cloud-sdk/completion.zsh.inc"
fi

# Python paths
export PATH="$HOME/Library/Python/3.12/bin:$PATH"

# Docker completions
if [ -d "$HOME/.docker/completions" ]; then
    fpath=($HOME/.docker/completions $fpath)
    autoload -Uz compinit
    compinit
fi

# Project-specific configuration
if [ -f "$(pwd)/.zshrc.devqai" ]; then
    source "$(pwd)/.zshrc.devqai"
elif [ -f "$(git rev-parse --show-toplevel 2>/dev/null)/.zshrc.devqai" ]; then
    source "$(git rev-parse --show-toplevel)/.zshrc.devqai"
fi

# Function to display environment info
function show_env_vars() {
    echo "USER: $USER"
    echo "HOME: $HOME"
    echo "SHELL: $SHELL"
    echo "PWD: $PWD"
    echo "EDITOR: $EDITOR"
    echo "VIRTUAL_ENV: $VIRTUAL_ENV"
}

# Enhanced zoxide aliases
alias zz="z -"
alias zi="zi"
alias zq="zoxide query"
alias zr="zoxide remove"

# Git shortcuts
alias gs="git status"
alias gp="git pull"
alias gc="git commit -m"
alias gd="git diff"
alias glog="git log --oneline --graph --decorate"

echo "⚡ Zed terminal ready for DevQ.ai development!"
EOF

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Configure Zed to use the new .zshrc file:"
echo "   • Open Zed settings (Cmd+,)"
echo "   • Go to Terminal settings"
echo "   • Set shell to: /bin/zsh"
echo "   • Set shell arguments to: -l -c 'source ~/.zshrc.zed; exec zsh'"
echo ""
echo "2. Or manually source the configuration in any terminal:"
echo "   source ~/.zshrc.zed"
echo ""
echo "3. Navigate to your DevQ.ai project and run:"
echo "   source .zshrc.devqai"
echo ""
echo "🎉 Your Zed terminal will now have:"
echo "   • Powerlevel10k theme"
echo "   • zoxide for smart directory navigation"
echo "   • Enhanced autosuggestions and syntax highlighting"
echo "   • DevQ.ai project-specific aliases and functions"
echo "   • All your existing tools and configurations"
