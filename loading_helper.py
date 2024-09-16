import numpy as np
import os
import torch
import nibabel as nib
import shutil
import multiprocessing
def generateMyTrainingData(args,ls):
    all_nodis_idx = 0
    scan_counter = 0
    for scan_folder in ls:
    # for scan_folder in os.listdir(args.training_data_path):
        scan_counter += 1
        print("Creating \"my_training_data\" with custom preprocessed scan patches,  at scan: " + str(
            scan_counter) + " of " + str(len(ls)))
        scan_vol = nib.load(args.training_data_path + "/" + scan_folder + "/image_total.nii").get_fdata()
        path="../data/my_training_data/" + scan_folder + "_" + str(scan_vol.shape)
        if os.path.isdir(path):
            # print("continue")
            # continue
            shutil.rmtree(path)
        os.makedirs(path)
        scan_segm = np.zeros_like(scan_vol)
        nodule_mean_centroids = np.empty((0,3))
        for nodule_folders in os.listdir(args.training_data_path + "/" + scan_folder):
            if os.path.isdir(args.training_data_path + "/" + scan_folder + "/" + nodule_folders):
                nodule_anni_centroids = np.empty((0,3))
                for nodule_annotation_folders in os.listdir(
                        args.training_data_path + "/" + scan_folder + "/" + nodule_folders):
                    nod_anni_mask = nib.load(
                        args.training_data_path + "/" + scan_folder + "/" + nodule_folders + "/" + nodule_annotation_folders + "/mask.nii").get_fdata().astype(
                        int)
                    nod_anni_bbox = np.loadtxt(
                        args.training_data_path + "/" + scan_folder + "/" + nodule_folders + "/" + nodule_annotation_folders + "/bbox.txt",
                        delimiter=',')
                    nod_anni_bbox = (slice(int(nod_anni_bbox[0, 0]), int(nod_anni_bbox[0, 1])),
                                     slice(int(nod_anni_bbox[1, 0]), int(nod_anni_bbox[1, 1])),
                                     slice(int(nod_anni_bbox[2, 0]), int(nod_anni_bbox[2, 1])))
                    scan_segm[nod_anni_bbox] += nod_anni_mask
        scan_segm[np.where(scan_segm > 0)] = 1
        if np.max(scan_segm) < 1:
            print("This scan has max segm val: " + str(np.max(scan_segm)))
            continue
        for i in range(scan_vol.shape[-1]):
            scan_vol_slice=np.array(scan_vol[:,:,i],dtype=np.float16)
            seg_slice=np.array(scan_segm[:,:,i],dtype=np.float16)
            # if np.max(scan_vol_slice)==0:continue
            # print(f"scan_vol_slice.shape={scan_vol_slice.shape}")
            np.save(path+ "/img_" + str(i), scan_vol_slice)
            np.save(path + "/seg_" + str(i), seg_slice)
        # scan_counter += 1

def generateMyTestingData(args,ls):
    scan_counter = 0
    for scan_folder in ls:
    # for scan_folder in os.listdir(args.testing_data_path):
        all_patches_idx = 0
        print("Creating \"my_testing_data\" with custom preprocessed scan patches, at scan: " + str(
            scan_counter) + " of " + str(len(ls)))
        scan_vol = nib.load(args.testing_data_path + "/" + scan_folder + "/image_total.nii").get_fdata()
        path="../data/my_testing_data1/" + scan_folder + "_" + str(scan_vol.shape)
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)
        for i in range(scan_vol.shape[-1]):
            scan_vol_slice=np.array(scan_vol[:,:,i],dtype=np.float16)
            # if np.max(scan_vol_slice)==0:continue
            # print(f"scan_vol_slice.shape={scan_vol_slice.shape}")
            np.save(path+ "/img_" + str(i), scan_vol_slice)
        scan_counter += 1

def split(l,i,all_i):
    num_deal=int(len(l)/all_i)+1
    ls_i=l[i*num_deal:min((i+1)*num_deal,len(l))]
    if(i==0):print(ls_i)
    return ls_i
def worker(args,test_ls,train_ls):
    # 
    generateMyTrainingData(args,train_ls)
    # generateMyTestingData(args,test_ls)
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--patch_size', nargs='+', type=int, default=[64,64,64],help="Size of 3D boxes cropped out of CT volumes as model input")
    parser.add_argument('--training_data_path', type=str, help="Set the path to training dataset")
    parser.add_argument('--testing_data_path', type=str, help="Set the path to testing dataset")
    parser.add_argument('--testing_data_solution_path', type=str,
                        help="Set the path to solution of testing dataset")
    parser.add_argument('--lr', default=0.01, type=float, help="Learning rate")
    parser.add_argument('--train', type=bool, default=False, help="Use True for training")
    parser.add_argument('--test', type=bool, default=False, help="Use True for testing")
    parser.add_argument('--model_path', type=str, help="Set the path of the model to be tested")
    parser.add_argument('--num_classes', type=int, default=10, help="Number of classes / number of output layers")
    args = parser.parse_args()
    test_ls=os.listdir(args.testing_data_path)
    train_ls=os.listdir(args.training_data_path)
    all_i=32
    p_num=range(all_i)
    plist=[]
    for i in p_num:
        test_ls_i=split(test_ls,i,all_i)
        train_ls_i=split(train_ls,i,all_i)
        p = multiprocessing.Process(target=worker, args=(args,test_ls_i,train_ls_i))
        plist.append(p)
    for p in plist:
        p.start()
    for p in plist:
        p.join()
