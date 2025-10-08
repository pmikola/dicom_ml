import sys
import pickle
import onnxruntime as ort
import torch
import matplotlib
import os
import json
import shutil
import pydicom
from DepiModels import HairSkinClassifier

# matplotlib.use("TkAgg")

def save_models_and_scripts_to_pickle(root_dir, output_file='models_and_scripts.pkl'):
    """
    Save model folders (hc_model, skin_type_model) and Python scripts to a pickle file.
    
    Args:
        root_dir: Root directory containing the models and scripts
        output_file: Name of the output pickle file
    """
    def collect_files_from_dirs(base_path, target_dirs, py_files):
        """Collect all files from specified directories and Python files"""
        files_data = {}
        
        # Collect files from model directories
        for dir_name in target_dirs:
            dir_path = os.path.join(base_path, dir_name)
            if os.path.exists(dir_path):
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, base_path)
                        try:
                            with open(file_path, 'rb') as f:
                                files_data[rel_path] = f.read()
                            print(f"Added: {rel_path}")
                        except Exception as e:
                            print(f"Warning: Could not read {rel_path}: {e}")
        
        # Collect Python files
        for py_file in py_files:
            file_path = os.path.join(base_path, py_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        files_data[py_file] = f.read()
                    print(f"Added: {py_file}")
                except Exception as e:
                    print(f"Warning: Could not read {py_file}: {e}")
        
        return files_data
    
    # Define what to save
    model_dirs = ['hc_model', 'skin_type_model']
    python_files = ['DepiModels.py']
    
    print(f"Collecting files from {root_dir}...")
    files_data = collect_files_from_dirs(root_dir, model_dirs, python_files)
    
    # Create the payload
    payload = {
        'root_name': os.path.basename(root_dir),
        'files': files_data,
        'metadata': {
            'total_files': len(files_data),
            'model_dirs': model_dirs,
            'python_files': python_files,
            'created_at': str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
        }
    }
    
    # Save to pickle
    output_path = os.path.join(root_dir, output_file)
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nSuccessfully saved {len(files_data)} files to: {output_path}")
        print(f"Total size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        return True
    except Exception as e:
        print(f"Error saving pickle file: {e}")
        return False

def restore_from_pickle(pickle_file, output_dir=None):
    try:
        with open(pickle_file, 'rb') as f:
            payload = pickle.load(f)
        if output_dir is None:
            output_dir = f"restored_{payload['root_name']}"
        os.makedirs(output_dir, exist_ok=True)
        for rel_path, content in payload['files'].items():
            dest_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, 'wb') as f:
                f.write(content)
            print(f"Restored: {rel_path}")
        print(f"\nRestored {len(payload['files'])} files to: {output_dir}")
        return True
    except Exception as e:
        print(f"Error restoring from pickle: {e}")
        return False

# ####################### IMPORT DCM
base = 'C:\PRACA\Depi\Prg_prj\ml_dicom\dicom_ml'
pass_dicom1 = '2024-03-18-14-39-21_d5b95953-c78a-4cc4-b596-c65caa435990'
pass_dicom = pass_dicom1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
filename = pydicom.data.data_manager.get_files(base, pass_dicom + '.dcm')[0]
disp = 0
x = pydicom.dcmread(filename, force=True) # Note : read dcm file
print(f"DICOM shape: {x.pixel_array.shape}")
print(f"DICOM dtype: {x.pixel_array.dtype}")
# ####################### SAVE MODELS AND SCRIPTS TO PICKLE
save_success = save_models_and_scripts_to_pickle(base, 'HairSkinClassifier.pkl')
# ####################### RESTORE MODELS AND SCRIPTS FROM PICKLE
restore_success = restore_from_pickle('HairSkinClassifier.pkl')

model = HairSkinClassifier(disp,device)
model.eval()
print(device)
with torch.no_grad():
    raw_output,colors = model(x)
    print(raw_output,colors)
    sys.exit()
