import sys

import onnxruntime as ort

import torch
import matplotlib
import os
import json
import shutil
from pydicom.data import get_testdata_file
import pydicom
from transformers import ViTForImageClassification
from DepiModels import HairSkinClassifier
from torch.package import PackageExporter
from torch.package import PackageImporter
import pkg_bootstrap

# matplotlib.use("TkAgg")
# ####################### IMPORT DCM
base = 'C:\PRACA\Depi\Prg_prj\ml_dicom\dicom_ml'
pass_dicom1 = '2024-03-18-14-39-21_d5b95953-c78a-4cc4-b596-c65caa435990'
pass_dicom = pass_dicom1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
filename = pydicom.data.data_manager.get_files(base, pass_dicom + '.dcm')[0]
disp = 0
x = pydicom.dcmread(filename, force=True) # Note : read dcm file
# model = HairSkinClassifier(disp,device)
# model.eval()
# print(device)

# def _add_dir_to_pkg(exporter: PackageExporter, dir_path: str, package_root: str, manifest: list) -> None:
#     base_name = os.path.basename(os.path.normpath(dir_path))
#     for root, _, files in os.walk(dir_path):
#         rel_dir = os.path.relpath(root, dir_path)
#         pkg = package_root + '.' + base_name
#         if rel_dir != '.':
#             pkg += '.' + rel_dir.replace(os.sep, '.')
#         for fname in files:
#             src = os.path.join(root, fname)
#             with open(src, 'rb') as fsrc:
#                 data = fsrc.read()
#             exporter.save_binary(pkg, fname, data)
#             manifest.append({
#                 "original_dir": base_name,
#                 "relative_path": (fname if rel_dir == '.' else os.path.join(rel_dir, fname)).replace('\\', '/'),
#                 "package": pkg,
#                 "resource": fname
#             })

# def _extract_dir_from_manifest(importer: PackageImporter, package_root: str, base_name: str, dest_dir: str, manifest: list) -> None:
#     os.makedirs(dest_dir, exist_ok=True)
#     entries = [e for e in manifest if e.get("original_dir") == base_name]
#     for e in entries:
#         target_path = os.path.join(dest_dir, e["relative_path"].replace('/', os.sep))
#         os.makedirs(os.path.dirname(target_path), exist_ok=True)
#         reader = importer.get_resource_reader(e["package"])  # stream to avoid loading entire file in memory
#         with reader.open_resource(e["resource"]) as src, open(target_path, 'wb') as dst:
#             shutil.copyfileobj(src, dst, length=1024 * 1024)
            
# with PackageExporter("HairSkinClassifier.pkg") as ex:
#     ex.extern("torch");         ex.extern("torch.**")
#     ex.extern("numpy");         ex.extern("numpy.**")
#     ex.extern("cv2");           ex.extern("cv2.**")
#     ex.extern("transformers");  ex.extern("transformers.**")
#     ex.mock("matplotlib")
#     ex.mock("matplotlib.*")
#     # Embed model directories and a manifest for extraction using packaged bootstrap
#     pkg_bootstrap.embed_model_dirs(ex, base)
#     ex.save_module("DepiModels")
#     ex.save_module("pkg_bootstrap")
#     ex.save_pickle("model","model.pkl", model.eval())


# torch.save(model, "HairSkinClassifier.pth")
#path = "HairSkinClassifier.pth"
#torch.serialization.add_safe_globals([HairSkinClassifier,ViTForImageClassification])
#model: HairSkinClassifier = torch.load(path,map_location="cpu",weights_only=False)
# model.eval()
with torch.no_grad():
    imp = PackageImporter("HairSkinClassifier.pkg")
    try:
        bootstrap = imp.import_module("pkg_bootstrap")
        bootstrap.extract_assets(imp, os.path.dirname(__file__))
    except Exception as e:
        print(f"Resource extraction skipped: {e}")
    model = imp.load_pickle("model", "model.pkl")
    model.eval()
    raw_output,colors = model(x)
    print(raw_output,colors)
    sys.exit()
