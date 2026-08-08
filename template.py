from pathlib import Path
import os

project_name = "src"

list_of_files=[
    #day1
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    "setup.py",
    "pyproject.toml",
    "requirements.txt",
    "demo.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        print(f"Creating directory: {filedir} for the file: {filename}")
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass
            print(f"Creating empty file: {filepath}")
    else:
        print(f"{filename} already exists")