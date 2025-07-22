import sys
import torch
import matplotlib
from pydicom.data import get_testdata_file
import pydicom
from transformers import ViTForImageClassification

from model import HairSkinClassifier

matplotlib.use("TkAgg")
####################### IMPORT DCM
base = 'C:\PRACA\Depi\Prg_prj\ml_dicom\dicom_ml'
pass_dicom1 = '2024-03-18-14-39-21_d5b95953-c78a-4cc4-b596-c65caa435990'
pass_dicom = pass_dicom1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
filename = pydicom.data.data_manager.get_files(base, pass_dicom + '.dcm')[0]
disp = 0
model = HairSkinClassifier(disp,device)
torch.save(model, "HairSkinClassifier.pth")

x = pydicom.dcmread(filename, force=True) # Note : read dcm file
path = "HairSkinClassifier.pth"
torch.serialization.add_safe_globals([HairSkinClassifier,ViTForImageClassification])
model: HairSkinClassifier = torch.load(path,map_location="cpu",weights_only=False)
model.eval()
raw_output,colors = model(x)
print(raw_output,colors)
sys.exit()
