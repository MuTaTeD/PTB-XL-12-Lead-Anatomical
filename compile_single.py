import os
import re
import subprocess

# 1. Read main.tex
with open('manuscript/main.tex', 'r') as f:
    content = f.read()

# 2. Modify to single column format and fix margins
content = content.replace('\\documentclass[5p,twocolumn,10pt]{elsarticle}', '\\documentclass[12pt]{elsarticle}\n\\usepackage[margin=1in]{geometry}')

# 3. Comment out graphical abstract
content = re.sub(r'(\\begin\{graphicalabstract\}.*?\\end\{graphicalabstract\})', lambda m: '\n'.join(['% ' + line for line in m.group(1).split('\n')]), content, flags=re.DOTALL)

# 4. Comment out highlights
content = re.sub(r'(\\begin\{highlights\}.*?\\end\{highlights\})', lambda m: '\n'.join(['% ' + line for line in m.group(1).split('\n')]), content, flags=re.DOTALL)

# 5. Write to main_single.tex
with open('manuscript/main_single.tex', 'w') as f:
    f.write(content)

# 6. Compile
print("Compiling main_single.tex...")
subprocess.run(['/home/awais/bin/pdflatex', '-interaction=nonstopmode', 'main_single.tex'], cwd='manuscript')
subprocess.run(['/home/awais/bin/bibtex', 'main_single'], cwd='manuscript')
subprocess.run(['/home/awais/bin/pdflatex', '-interaction=nonstopmode', 'main_single.tex'], cwd='manuscript')
subprocess.run(['/home/awais/bin/pdflatex', '-interaction=nonstopmode', 'main_single.tex'], cwd='manuscript')

# 7. Move PDF
os.rename('manuscript/main_single.pdf', 'manuscript/main_single_column_review.pdf')
print("Successfully generated main_single_column_review.pdf")
