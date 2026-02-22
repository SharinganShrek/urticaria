# Push changes to repo

Run these in Git Bash (or any terminal where `git` works) from the project root:

```bash
cd ~/Documents/GitHub/urticaria-clone

# See what changed
git status

# Stage all changes
git add -A

# Commit with a descriptive message
git commit -m "Add speaker type & gender labeling, keyword-based topic analysis, BERTopic setup guide"

# Push to remote (default branch, usually main)
git push
```

If your branch is not `main`:
```bash
git push origin $(git branch --show-current)
```

If you need to set upstream on first push:
```bash
git push -u origin main
```
