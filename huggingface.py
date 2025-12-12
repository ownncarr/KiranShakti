from huggingface_hub import login, upload_folder

# Push your model files
upload_folder(folder_path=".", repo_id="OwnnCarr/pv_detector", repo_type="model")
