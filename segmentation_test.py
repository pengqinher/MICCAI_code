import os
import shutil
import torch
from torchmetrics import Dice
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from monai.losses.dice import DiceLoss
diceloss=DiceLoss(reduction='none')
from efficientvit.seg_model_zoo import create_seg_model
print("import done")

def deal_input(image,im_max=1000,rgb_max=255):
    image[image < -im_max] = -im_max
    image[image > im_max] = im_max
    image = (image - (-im_max)) / (im_max - (-im_max))
    # image=image-image.min()
    image = image*255/(2*im_max)
    return image
def deal_output(output,s):
    output = (
        F.interpolate(output, size=(s[0],s[1]), mode="bilinear")
        # F.interpolate(output[:, :, i], size=(image.shape[2], image.shape[3]), mode="bilinear")
    )
    output[output>0]=1.
    output[output<=0]=0.
    return output

def test_fnc_final(test_model, args):
    num=0

    test_model=test_model.cuda()
    test_model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        with torch.no_grad():
            for scan_folder in os.listdir(args.testing_data_path):
                print(scan_folder)
                sl=scan_folder.split('_')
                scan_id=int(sl[sl.index("scan")+1])
                path=os.path.join(args.test_pred_path,("scan_" + str(scan_id)))
                if os.path.isdir(path):
                    # print("continue")
                    # continue
                    print("path exist")
                    shutil.rmtree(path)
                os.makedirs(path)
                scan_val = nib.load(args.testing_data_path + "/" + scan_folder + "/image_total.nii").get_fdata()
                # print(f"scan_val.type={type(scan_val)}")
                # print(f"scan_val.shape={scan_val.shape}")
                scan_val=np.array(scan_val,dtype=np.float16)
                scan_val=scan_val.transpose(2,0,1)
                scan_val=np.expand_dims(scan_val,axis=1)
                scan_val=np.concatenate((scan_val,scan_val,scan_val),axis=1)
                im_b=torch.tensor(scan_val,dtype=torch.float16).cuda()

                im_b=deal_input(im_b)

                preds=[]
                for pi in np.arange(start=0, stop=im_b.shape[0], step=args.batch_size):
                    if (pi + args.batch_size) >= im_b.shape[0]:
                        pred_pi_seg = test_model(im_b[pi:])
                    else:
                        pred_pi_seg = test_model(im_b[pi:pi + args.batch_size])
                    preds.append(pred_pi_seg)
                # preds=torch.asarray(preds)
                preds = torch.cat(preds, dim=0)
                print(f"type(preds)={type(preds)}")
                print(f"preds.shape={preds.shape}")
                output=deal_output(preds,[im_b.shape[2],im_b.shape[3]])
                print(f"output.shape={output.shape}")
                output=np.squeeze(np.array(output.cpu(),dtype=np.int8),axis=1)
                output_seg=output.transpose(1,2,0)
                pred_seg_nii = nib.Nifti1Image(output_seg, affine=np.eye(4))
                nib.save(pred_seg_nii,
                        os.path.join(args.test_pred_path,"scan_" + str(scan_id) + "/prediction_total.nii"))
                print("saved prediction " + str(num))



def calculateDice(args):
    dc = 0.0
    dl=0.0
    counter = 0
    dice = Dice(average='micro',ignore_index=0)
    for scan_file in os.listdir(args.testing_data_solution_path):
        y_mask = torch.from_numpy(
            nib.load(args.testing_data_solution_path + "/" + scan_file + "/segmentation_total.nii").get_fdata()).int()
        pred_seg = torch.from_numpy(
            nib.load(os.path.join(args.test_pred_path, scan_file + "/prediction_total.nii")).get_fdata())
        print(f"y:{y_mask.shape}")
        print(f"y:{y_mask.max()}")
        print(f"p:{pred_seg.shape}")
        print(f"p:{pred_seg.min()}")
        print("Dice score of " + str(scan_file) + ": " + str(dice(pred_seg, y_mask).item()))
        print(f"diceloss={diceloss(pred_seg,y_mask).mean()}")
        dc += dice(pred_seg, y_mask)
        dl+=diceloss(pred_seg,y_mask).mean()
        counter += 1

    return dc / counter,dl/counter
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--testing_data_path', type=str, help="Set the path to testing dataset")
    parser.add_argument('--testing_data_solution_path', type=str,
                        help="Set the path to solution of testing dataset")
                        
    parser.add_argument('--test_pred_path', type=str,
                        help="Set the path to prediction of testing dataset")
    parser.add_argument('--model_path', type=str, help="Set the path of the model to be tested")
    args = parser.parse_args()
    model = create_seg_model("l1", "med_seg", weight_url=args.model_path)
    # test_fnc_final(model, args)
    print(f"final_result={calculateDice(args)}")