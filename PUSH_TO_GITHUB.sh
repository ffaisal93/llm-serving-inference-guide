#!/bin/bash
# Quick script to push to GitHub
# Run: bash PUSH_TO_GITHUB.sh

set -e

echo "🚀 Setting up GitHub repository..."

# Initialize git if not already done
if [ ! -d .git ]; then
    echo "📦 Initializing git repository..."
    git init
fi

# Add all files
echo "📝 Adding files..."
git add .

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: Complete LLM serving and inference learning guide" || echo "No changes to commit"

# Add remote (if not already added)
if ! git remote | grep -q origin; then
    echo "🔗 Adding remote repository..."
    git remote add origin https://github.com/ffaisal93/llm-serving-inference-guide.git
else
    echo "✅ Remote already configured"
fi

# Set branch to main
git branch -M main

echo ""
echo "✅ Ready to push!"
echo ""
echo "Next steps:"
echo "1. Create repository on GitHub: https://github.com/new"
echo "   - Name: llm-serving-inference-guide"
echo "   - Description: Complete learning guide for LLM serving and inference in production"
echo "   - Don't initialize with README, .gitignore, or license"
echo ""
echo "2. Then run:"
echo "   git push -u origin main"
echo ""
echo "Or run this script again after creating the repository."

