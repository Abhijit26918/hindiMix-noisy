#!/usr/bin/env bash
# ─────────────────────────────────────────────────────
# daily_push.sh — Run this every evening before you sleep
# Usage: bash daily_push.sh "what you did today"
# ─────────────────────────────────────────────────────

DATE=$(date +"%Y-%m-%d")
MSG="${1:-Daily update $DATE}"

echo "📦 Staging all changes..."
git add -A

# Check if there's anything to commit
if git diff --cached --quiet; then
  echo "⚠️  Nothing new to commit today. Did you forget to save your work?"
  exit 0
fi

echo "💬 Committing: $MSG"
git commit -m "[$DATE] $MSG"

echo "🚀 Pushing to GitHub..."
git push origin master

echo "✅ Done! Today's work is on GitHub."
echo "   Keep going — $(( ($(date -d '2026-04-25' +%s) - $(date +%s)) / 86400 )) days left until April 25."
