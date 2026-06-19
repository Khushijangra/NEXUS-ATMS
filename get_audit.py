import subprocess
import json

def run(cmd):
    return subprocess.check_output(cmd, shell=True, text=True).strip()

remotes = run('git remote -v')
branch = run('git branch --show-current')

try:
    deleted = len([l for l in run('git status --short').split('\n') if l.startswith(' D') or l.startswith('D ')])
    added = len([l for l in run('git status --short').split('\n') if l.startswith('??') or l.startswith('A')])
    modified = len([l for l in run('git status --short').split('\n') if l.startswith(' M') or l.startswith('M ')])
except:
    deleted, added, modified = 0, 0, 0

print(json.dumps({
    "remote": remotes,
    "branch": branch,
    "deleted": deleted,
    "added": added,
    "modified": modified
}, indent=2))
