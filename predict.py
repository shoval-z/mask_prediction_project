import os
import argparse
import pandas as pd
from dataset import mask_dataset
from darknet53_yoloV3 import Darknet53, residual_block, eval_only
from utils import *
import gdown

# creating the csv
def make_prediction(model, device, test_loader,test_dataset):
    data_dict = {'filename': [], 'x':[], 'y':[], 'w':[],'h':[], 'proper_mask':[]}
    print('creating csv')
    model.eval()
    with torch.no_grad():
        for idx,(x, y_bb, y_class,origin_size) in enumerate(test_loader):
            origin_size = origin_size.squeeze(1).to(device)
            x = x.to(device).float()
            out_class, out_bb = model(x)
            out_bb = torch.mul(out_bb, origin_size)
            out_bb = out_bb.cpu().numpy()
            pred = (out_class > 0.5).cpu().numpy()
            data_dict['x'].extend(out_bb[:, 0])
            data_dict['y'].extend(out_bb[:, 1])
            data_dict['w'].extend(out_bb[:, 2])
            data_dict['h'].extend(out_bb[:, 3])
            data_dict['proper_mask'].extend(pred)
    data_dict['filename'] = test_dataset.image_id
    return data_dict


parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('path', type=str, help='tsv file path')
args = parser.parse_args()
module_path = os.path.dirname(os.path.realpath(__file__))
# gdrive_file_id = '1q21vaR_OJmjW2TNSWbQBWF4hzwCpWv_v'
gdrive_file_id = '17SXNA_P7xbuas5zY0zBNZVWRT_Jb44Nk'
url = f'https://drive.google.com/uc?id={gdrive_file_id}'
model_path = os.path.join(module_path, 'model.pth')
gdown.download(url, model_path, quiet=False)

# Load Trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model.pth')
model.to(device)

# create test loader
test_dataset = mask_dataset(dataset='test', path=args.path)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False,
                                              num_workers=4,
                                              pin_memory=True)

# create csv and return measures
data_dict = make_prediction(model, device, test_loader,test_dataset)
prediction_df = pd.DataFrame(data_dict)
prediction_df.to_csv("prediction.csv", index=False, header=True)
eval_only(model, device, test_loader, test_dataset, acc_loss_weight=1.)