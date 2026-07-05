import os
import re
from pathlib import Path

project_root = Path(__file__).resolve().parents[0]
path = project_root / "v2" / "final_dissertation_manuscript" / "paper_complete_revised.tex"

with open(path, "r", encoding="utf-8") as f:
    tex = f.read()

# Replace document class
tex = tex.replace(r"\documentclass{ieeeaccess}", r"\documentclass[journal]{IEEEtran}")

# Remove ieeeaccess specific macros
tex = tex.replace(r"\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}", "")
tex = tex.replace(r"\doi{10.1109/ACCESS.2023.0322000}", "")
tex = tex.replace(r"\titlepgskip=-15pt", "")
tex = tex.replace(r"\begin{keywords}", r"\begin{IEEEkeywords}")
tex = tex.replace(r"\end{keywords}", r"\end{IEEEkeywords}")

# Fix author block to standard IEEEtran
old_author = r"""\author{\uppercase{First A. Author}\authorrefmark{1},
\uppercase{Second B. Author\authorrefmark{2}, and Third C. Author\authorrefmark{3}}}
\address[1]{Department of Computer Science, University of Technology, City, Country}
\tfootnote{This work was supported in part by the National Science Foundation under Grant XXXX.}

\markboth
{Author \headeretal: A Unified Semantic Predictive Graph Reinforcement Learning Framework}
{Author \headeretal: A Unified Semantic Predictive Graph Reinforcement Learning Framework}

\corresp{Corresponding author: First A. Author (e-mail: author@university.edu).}"""

new_author = r"""\author{First A. Author,
Second B. Author, and Third C. Author
\thanks{This work was supported in part by the National Science Foundation under Grant XXXX.}
\thanks{First A. Author is with the Department of Computer Science, University of Technology, City, Country (e-mail: author@university.edu).}}

\markboth{Author \MakeLowercase{\textit{et al.}}: A Unified Semantic Predictive Graph Reinforcement Learning Framework}{}
"""
tex = tex.replace(old_author, new_author)

# IEEEtran requires \maketitle BEFORE \begin{abstract}
tex = tex.replace(r"\maketitle", "")
tex = re.sub(r"(\\begin\{abstract\})", r"\\maketitle\n\n\1", tex)

with open(path, "w", encoding="utf-8") as f:
    f.write(tex)
print("Updated to IEEEtran.")
