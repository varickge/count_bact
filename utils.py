import os
import cv2
import json
import torch
import pickle
import torchvision
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches

colors = {'white':      "\033[1;37m",
          'yellow':     "\033[1;33m",
          'green':      "\033[1;32m",
          'blue':       "\033[1;34m",
          'cyan':       "\033[1;36m",
          'red':        "\033[1;31m",
          'magenta':    "\033[1;35m",
          'black':      "\033[1;30m",
          'darkwhite':  "\033[0;37m",
          'darkyellow': "\033[0;33m",
          'darkgreen':  "\033[0;32m",
          'darkblue':   "\033[0;34m",
          'darkcyan':   "\033[0;36m",
          'darkred':    "\033[0;31m",
          'darkmagenta':"\033[0;35m",
          'darkblack':  "\033[0;30m",
          'off':        "\033[0;0m"}

def get_boxes_from_yolo(file_path,image_shape, get_scores=False):
    '''
    Read the yolo labels from the given file, convert to xxyy format and return
    Inputs:
        - `file_path`: Path to a .txt label file
        - `image_shape`: Shape of the image for deconverting
        - `get_scores`: If set to true, returns the scores of the boxes too (For deconverting predictions)
    Output:
        - if `get_scores == True`: N x 5 bboxes of [x1 y1 x2 y2 score] format
        - if `get_scores == False`: N x 4 bboxes of [x1 y1 x2 y2] format
    '''
    with open(file_path,"r") as f:
        lines = [line.replace("\n","").split() for line in f.readlines()]
    
    res=[]
    scores = []
    for line in lines:
        line = [float(i) for i in line]
        xmin = (line[1]-line[3]/2)*image_shape[1]
        xmax = (line[1]+line[3]/2)*image_shape[1]
        ymin = (line[2]-line[4]/2)*image_shape[0]
        ymax = (line[2]+line[4]/2)*image_shape[0]
        if get_scores:
            scores.append(line[5])
        res.append((xmin,ymin,xmax,ymax))
    
    res = np.array(res).astype(np.int16)
    if get_scores:
        return res, np.array(scores)
    
    return res

def get_boxes_from_yolo_folder(folder_path,image_shape):
    '''
    Finds all .txt labels for yolov5 in the given folder and deconverts those to [x1 y1 x2 y2] format.
    Inputs:
        - `folder_path`: A path to a folder containing yolo labels in .txt format
        - `image_shape`: Shape of the image according to which the coordinates should be deconverted
    Output:
        - `N x 4 bounding boxes of [x1 y1 x2 y2] format`
    '''
    
    # Get .txt file paths in the given folder_path
    label_paths = glob(os.path.join(folder_path,"*.txt"))
    if len(label_paths)==0:
        raise ValueError(f"{folder_path} does not contain any .txt files")
    
    # A list for storing all deconverted boxes
    all_boxes = []

    # Deconvert labels from each file and add to all_boxes
    for path in label_paths:
        boxes= get_boxes_from_yolo(path,image_shape)
        all_boxes.extend(boxes)
   
    # Stack the boxes into a torch.Tensor
    all_boxes = torch.from_numpy(np.array(all_boxes)).to(int)

    return all_boxes
    
    
def rectangles_on_mip(mip, boxes, box_format="xxyy_sep", color=(0,0,255), thickness=1, show_indices=True, text_color=(255,255,255), text_size=0.3):
    '''
    A function for ploting rectangles on a picture, returns the picture\n
    Parameters:
        - `mip`: The picture(ndarray) to which the rectangles will be applied to
        - `boxes`: list of boxes, which will be applied on the image
        - `box_format`: the format of the boxes which are given to the function
        Accepts one of the following values:
            - `xyxy_sep`: if the boxes are given in the following format `([xmins...], [ymins...], [xmaxs...], [ymaxs...])` |-> each coordinates of boxes are in the same lists
            - `xxyy_sep`: if the boxes are given in the following format `([xmins...], [xmaxs...], [ymins...], [ymaxs...])` |
            - `xyxy_one`: if the boxes are given in the following format `[xmin, ymin, xmax, ymax], ...` |-> each box is separate
            - `xxyy_one`: if the boxes are given in the following format `[xmin, xmax, ymin, ymax], ...` |
            - `xywh`: if the boxes are given in the following format `(xcenter, ycenter, width, height)` 
            - `pt1,pt2`: if the boxes are given in the following format `((xmin, ymin), (xmax, ymax))`
        - `color`: an RGB `color(R,G,B)` : `R,G,B <-- (0,255)`
    '''
    box_format = box_format.lower()
    im = mip.copy()
    if box_format=="xyxy_sep":
        boxes_for_plotting = [((boxes[0][i],boxes[1][i]),(boxes[2][i],boxes[3][i])) for i in range(len(boxes[0]))]
    elif box_format=="xxyy_sep":
        boxes_for_plotting = [((boxes[0][i],boxes[2][i]),(boxes[1][i],boxes[3][i])) for i in range(len(boxes[0]))]
    elif box_format=="xyxy_one":
        boxes_for_plotting = [((boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3])) for i in range(len(boxes))]
    elif box_format=="xxyy_one":
        boxes_for_plotting = [((boxes[i][0],boxes[i][2]),(boxes[i][1],boxes[i][3])) for i in range(len(boxes))]
    elif box_format=="pt1,pt2":
        boxes_for_plotting = boxes
    else:
        raise ValueError(f'`box_format` should be one of the following ["cr","xyxy_sep","xxyy_sep","xyxy_one","xxyy_one", "pt1,pt2"] but got {box_format}')
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i,box in enumerate(boxes_for_plotting):
        cv2.rectangle(im, box[0], box[1], color=color, thickness=1)
        if (show_indices):
            im = cv2.putText(im, f"{i}", box[0], font, text_size, text_color, 1, cv2.LINE_AA)    
    return im


def get_boxes_from_json_files(root_path, nms_threshold=2):
    '''
    Finds all .json files and deconverts those to boxes of [x1 y1 x2 y2] format.
    Inputs:
        - `root_path`: A path to a folder containing yolo labels in .txt format
        - `nms_threshold`: Threshold for excluding overlaping boxes. A value in between [0,1] if >1, nms will not be completed.
    Output:
        - `N x 4 bounding boxes of [x1 y1 x2 y2] format`
    '''
    
    # Get json filenames
    if root_path[-5:]!=".json":
        files = glob(os.path.join(root_path,"*.json"))
    else:
        files = [root_path]
    frames = []
    # Read all json files in root_path
    for file in files:
        with open(file, "r") as f:
            labels = json.load(f)
            frames.append([item["points"] for item in labels["shapes"]])
    
    # Collecting all boxes from all files
    all_boxes = []
    for index, _ in enumerate(frames):
        boxes = []
        for jindex, _ in enumerate(frames[index]):
            boxes.append(frames[index][jindex][0][:])
            boxes[-1].extend(frames[index][jindex][1][:])
        if boxes == []:
            continue
    
        all_boxes.append(boxes[:])
    
    boxes_per_frames = []
    for frame in all_boxes:
        boxes_per_frames.append(torch.tensor(frame))
       
    # Doing NMS. 
    resulting_boxes = boxes_per_frames[0]
    for boxes in boxes_per_frames[1:]:
        if boxes.ndim == 1:
            boxes = boxes[None, ...]
        iou_matrix = torchvision.ops.box_iou(resulting_boxes, boxes)
        new_box_indices = torch.where(iou_matrix.sum(dim=0)< nms_threshold)
        resulting_boxes = torch.cat((resulting_boxes, boxes[new_box_indices])) 
    
    return resulting_boxes[:,[0,2,1,3]]


class Metrics:
    '''
    A class for computing custom metrics
    '''
    def box_iom(self, boxes1: torch.Tensor, boxes2: torch.Tensor):
        '''
        Computes the intersection over minimum for each box in boxes1 with boxes2.
        Returns an intersection over minimum matrix of size N x M, where N is the count of boxes1 and M is the count of boxes2
        Inputs:
            - `boxes1`: N x 4 torch.Tensor
            - `boxes2`: M x 4 torch.Tensor
        Output:
            - `Matrix of IoMs of N x M shape`
        '''
        # Computing areas of the boxes
        area1 = torchvision.ops.box_area(boxes1)
        area2 = torchvision.ops.box_area(boxes2)

        # Getting the maximums of left-top coordinates
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        
        # Getting the minimums of right-bottom coordinates
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
        
        # Getting widths and heights of the intersections
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        
        # Computing intersection areas
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        # Getting the minimum areas for each pair of boxes (Broadcasting area1 M times for area2)
        min_area = torch.min(area1[:,None,...],area2)

        # Return the intersections divided by the minimum areas
        return inter/min_area


    def compare_boxes(self, true_boxes, pred_boxes, print_=False, dec_p=2, confusion_matrix=False):
        '''
        A method for comparing Ground Truth boxes with the predictions:
        Inputs:
            - `true_boxes`: N x 4 ground-truth boxes
            - `pred_boxes`: M x 4 predicted boxes
            - `print_`: If set to `True`, comparison will be printed
            - `dec_p`: The maximum decimal points for comparisons
            - `confusion_matrix`: If is set to `True`, confusion matrix for comparisons will be printed too
        Outputs:
            A tuple containing the following values:
                - `Precision`
                - `Recall`
                - `F1`
                - (`True Positives`, `False Negatives`, `False Positives`)
                - `Count Accuracy`
        '''
        iom = self.box_iom(true_boxes,pred_boxes)

        tp = 0
        fn = (iom.sum(dim=1)==0).sum()
        fp = (iom.sum(dim=0)==0).sum()

        # for each gt box check
        for i, row in enumerate(iom):
            if row.sum()!=0:
                col=row.argmax()
                iom[:,col]=0
                fn += (iom[i:, iom[i]>0]>0).sum()==1
                tp+=1

        fp += (iom.sum(dim=0)>0).sum()


        Precision = (tp/(tp+fp)).item()
        Recall = (tp/(tp+fn)).item()
        F1 = (2*Precision*Recall/(Precision+Recall)) if Precision+Recall>0 else 0


        if print_:
            if confusion_matrix:
                print(f"TP={tp}, FN={fn}, TN=Undefined, FP={fp}\n")
            print(f"{'Precision:':<11} {colors['darkgreen']}{round(Precision*100, dec_p)} %{colors['off']}\n{'Recall:':<11} {colors['darkgreen']}{round(Recall*100, dec_p)} %{colors['off']}\n{'F1:':<11} {colors['darkgreen']}{round(F1*100, dec_p)} %{colors['off']}")
        
        Accuracy = (1-abs(pred_boxes.shape[0]-true_boxes.shape[0])/true_boxes.shape[0])
        
        return Precision, Recall, F1, (tp,fn,fp), Accuracy
        

    def pretty_print(self, tens):
        '''
        A method for fully printing given tensor with row and column indexes
        Input:
            - `tens`: A torch.Tensor of `N x M shape`
        '''
        print("   ", end="")
        for i in range(tens.shape[1]):
            print(f"{i:<3}| ",end="")
        print("\n","-"*70)
        for r,i in enumerate(tens):
            print(f"{r})", end = " ")
            for j in i:
                print(f"{j.item():.2f}", end=" ")
            print("\n")
            
def print_stats(true_boxes, pred_boxes, times=None):
    '''
    A function for printing the comparisons of given boxes and other statistical data
    Inputs:
        - `true_boxes`: N x 4 ground-truth boxes
        - `pred_boxes`: M x 4 predicted boxes
        - `times`: `None` or a dict, containing 'total_time' keyword with a numeric value.
    Output:
        Prints all comparisons between given boxes.
    '''
    Metric = Metrics()
    
    accuracy = Metric.compare_boxes(true_boxes, pred_boxes, print_=False)[-1]*100
    print(f"Counting accuracy: {colors['darkgreen']} {accuracy:.2f}%{colors['off']}\n")
    print(f"\tActual bacteria count:    {colors['darkgreen']}{true_boxes.shape[0]}{colors['off']}")
    print(f"\tPredicted bacteria count: {colors['darkgreen']}{pred_boxes.shape[0]}{colors['off']}\n")
    if times:
        print(f"{'Detection time'} --- {colors['darkgreen']}{times['total_time']:.2f} sec.{colors['off']}\n")
    
    Metric.compare_boxes(true_boxes, pred_boxes, print_=False)[-1]*100
    
def show_detections(data_paths, pred_paths, model="new", true_box_format="yolo"):
    '''
    A function for showing the detections on their images:
    Inputs:
        - `data_paths`: list of paths for original data with Ground Truth labels
        - `pred_paths`: list of paths for predictions (Folders containing .txt files with [x, y, w, h, score, class, frame] formatted boxes)
        - `model`: 
            - if is set to `new`, comparisons for new model will be shown
            - if is set to `old`, comparisons for old model will be shown
    Output:
        Prints comparisons for given boxes, on their images
    '''
    Metric = Metrics()

    metrics = {
        "Count Accuracy":[],
        "Precision": [],
        "Recall": [],
        "F1": []
    }
    
    # getting labels paths from the data_paths
    labels_paths = [os.path.join(i,"labels") for i in data_paths]
    
    
    # For reading times. Each model saves times differently, that's why 
    if model == "new":
        times = []
    elif model == "old":
        with open("./Old_Model/times.txt", "r") as f:
            times = [{'total_time':float(i.replace("\n","").split(": ")[1])} for i in f.readlines()]

    # For each data load predictions, plot on the mip, compare with ground truth
    for i in range(len(data_paths)):
        # Loading the mip
        mip = plt.imread(os.path.join(data_paths[i],"output_mip.jpeg"))

        # Loading the detection results
        if model.lower() == "new":
            with open(pred_paths[i],"rb") as f:
                pred_boxes, time = pickle.load(f)
                pred_boxes = torch.from_numpy(pred_boxes)[:,:4].to(torch.float)

        elif model.lower() == "old":
            with open(pred_paths[i],"rb") as f:
                pred_boxes = torch.from_numpy(np.load(pred_paths[i])[:,[0,2,1,3]])
                pred_boxes = pred_boxes[:,:4].to(torch.float)

        # If your ground_truth labels are in json format, you can load them with the below commented line
        if true_box_format.lower() == "json":
            true_boxes = get_boxes_from_json_files(labels_paths)[:,[0,2,1,3]]

        # If labels are in yolo format, use this line
        elif true_box_format.lower() == "yolo":
            true_boxes = get_boxes_from_yolo_folder(labels_paths[i], mip.shape[:-1])

        mip = rectangles_on_mip(mip, true_boxes.to(int).numpy(), box_format='xyxy_one',color=(255, 255, 255), show_indices=False)
        mip = rectangles_on_mip(mip, pred_boxes.to(int).numpy(), box_format='xyxy_one',color=(255, 0, 0), show_indices=False)
        sup_t = f"Folder path: {data_paths[i]}"

        if mip.shape[0]>mip.shape[1]:
            mip = mip.transpose(1,0,2)
            sup_t = f"{sup_t}\n(Rotated 90°)"
        
        # Creating Legend 
        gt = patches.Patch(color='white', label='Ground Truth')
        pr = patches.Patch(color='red', label='Prediction')

        plt.figure(figsize=(40,40))
        plt.suptitle(sup_t, x = 0.5, y=0.76, fontsize=40, fontweight="bold")
        plt.legend(handles=[gt, pr],prop={'size': 30}, loc="upper right")

        plt.imshow(mip)
        plt.show()

        # Unpacking precision, recall, f1, (tn, fn, fp), count_accuracy
        p,r,f,_,a = Metric.compare_boxes(true_boxes, pred_boxes)

        # Multiplying by 100 for percentage, inserting into the metrics
        metrics["Count Accuracy"].append(a*100)
        metrics["Precision"].append(p*100)
        metrics["Recall"].append(r*100)
        metrics["F1"].append(f*100)

        
        if model == "new":
            times.append(time)
            
        print_stats(true_boxes, pred_boxes, times[i])

    # Returning metrics and times
    return metrics,times
