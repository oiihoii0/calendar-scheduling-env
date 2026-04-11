import os

file_path = r'c:\Users\obito\OneDrive\Desktop\Scheduling\app.py'

# Replacements to fix the corrupted characters
replacements = {
    'dYY': '🚀',
    '?"': '—',
    'dYs?': '⚡',
    'dY"<': '📋',
    'dY\'s': '💚',
    'dYZ_': '🎯',
    'dY".': '📅',
    'dY?': '🏠',
    '?,?': '⏱️',
    'dY"O': '🔌',
    '': '·',
    '+\'': '→',
    '?': '—',
    '\"': '—',
    '\'': '—',
}

with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

for old, new in replacements.items():
    content = content.replace(old, new)

# Fix specific manual lines that might be broken
content = content.replace('<title>CalendarSchedulingEnv  OpenEnv</title>', '<title>CalendarSchedulingEnv — OpenEnv</title>')
content = content.replace('#  Request / response models', '# ── Request / response models ──────────────────────────────────────────')
content = content.replace('#  Helpers', '# ── Helpers ───────────────────────────────────────────────────────────')
content = content.replace('#  Endpoints', '# ── Endpoints ─────────────────────────────────────────────────────────')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("File cleaned successfully.")
