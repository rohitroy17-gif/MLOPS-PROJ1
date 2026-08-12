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
    #day2
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/configuration/mongo_db_connection.py",
    f"{project_name}/data_access/proj1_data.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/pipline/training_pipeline.py"
    #day3
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/entity/estimator.py",
    "config/model.yaml",
    "config/schema.yaml",
    #day4
    f"{project_name}/configuration/aws_connection.py",
    f"{project_name}/cloud_storage/__init__.py",
    f"{project_name}/cloud_storage/aws_storage.py",
    f"{project_name}/entity/s3_estimator.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",
    #day5
    f"{project_name}/pipline/prediction_pipeline.py",
    "app.py"
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