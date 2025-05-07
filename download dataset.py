import kagglehub

# Download latest version
path = kagglehub.dataset_download("ashishjangra27/face-mask-12k-images-dataset")
print("Path to dataset files:", path)

"""
    Don't forget to set the Kaggle API token in your environment variables.
    You can do this by running the following command in your terminal:
    export KAGGLE_CONFIG_DIR=/path/to/your/kaggle.json
    Make sure to replace /path/to/your/kaggle.json with the actual path to your kaggle.json file.
    Alternatively, you can set the KAGGLE_CONFIG_DIR environment variable in your Python script:
    import os
    os.environ['KAGGLE_CONFIG_DIR'] = '/path/to/your/kaggle.json'
    You can also use the following command to verify if the environment variable is set correctly:
    print(os.environ.get('KAGGLE_CONFIG_DIR'))
   
"""