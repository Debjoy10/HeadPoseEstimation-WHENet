import numpy as np
import cv2
from whenet import WHENet
from utils import draw_axis
from tqdm import tqdm
import os
import argparse
import pandas as pd
from yolo_v3.yolo_postprocess import YOLO
from PIL import Image


def process_detection( model, img, bbox, args ):

    y_min, x_min, y_max, x_max = bbox
    # enlarge the bbox to include more background margin
    y_min = max(0, y_min - abs(y_min - y_max) / 10)
    y_max = min(img.shape[0], y_max + abs(y_min - y_max) / 10)
    x_min = max(0, x_min - abs(x_min - x_max) / 5)
    x_max = min(img.shape[1], x_max + abs(x_min - x_max) / 5)
    x_max = min(x_max, img.shape[1])

    img_rgb = img[int(y_min):int(y_max), int(x_min):int(x_max)]
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))
    img_rgb = np.expand_dims(img_rgb, axis=0)

    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,0,0), 2)
    yaw, pitch, roll = model.get_angle(img_rgb)
    yaw, pitch, roll = np.squeeze([yaw, pitch, roll])
    draw_axis(img, yaw, pitch, roll, tdx=(x_min+x_max)/2, tdy=(y_min+y_max)/2, size = abs(x_max-x_min)//2 )
#     print("yaw: {}".format(np.round(yaw)))
#     print("pitch: {}".format(np.round(pitch)))
#     print("roll: {}".format(np.round(roll)))
    
    if args.display == 'full':
        cv2.putText(img, "yaw: {}".format(np.round(yaw)), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
        cv2.putText(img, "pitch: {}".format(np.round(pitch)), (int(x_min), int(y_min) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
        cv2.putText(img, "roll: {}".format(np.round(roll)), (int(x_min), int(y_min)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
    return img, yaw, pitch, roll



def main(args):
    RECORD_IND_FRAME = True
    VIDEO_SRC_LIST = [0] if args.video == '' else args.video # if video clip is passed, use web cam
    yprdict = []
    whenet = WHENet(snapshot=args.snapshot)
    yolo = YOLO(**vars(args))
    
    for VIDEO_SRC in tqdm(VIDEO_SRC_LIST):
        if not RECORD_IND_FRAME:
            yaw = 0
            pitch = 0
            roll = 0
        else:
            yaw = []
            pitch = []
            roll = []
        n = 0
        
        cap = cv2.VideoCapture(VIDEO_SRC)
        print('cap info',VIDEO_SRC)
        ret, frame = cap.read()
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if args.output:
            out = cv2.VideoWriter(args.output, fourcc, 30, (frame.shape[1], frame.shape[0]))  # write the result to a video

        while True:
            try:
                ret, frame = cap.read()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                break
            n += 1
            img_pil = Image.fromarray(frame_rgb)
            bboxes, scores, classes = yolo.detect(img_pil)
            for bbox in bboxes:
                frame, y, p, r = process_detection(whenet, frame, bbox, args)
            # cv2.imshow('output',frame)
            if args.output:
                out.write(frame)
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord("q"):
            #     break
            if not RECORD_IND_FRAME:
                yaw += y
                pitch += p
                roll += r
                break
            else:
                yaw.append(y)
                pitch.append(p)
                roll.append(r)
        # Avg
        if not RECORD_IND_FRAME:
            yaw /= n
            pitch /= n
            roll /= n
            yprdict.append({
                'video': VIDEO_SRC,
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll
            })
        else:
            for i in range(len(yaw)):
                yprdict.append({
                    'video': VIDEO_SRC+'_frame{}'.format(i),
                    'yaw': yaw[i],
                    'pitch': pitch[i],
                    'roll': roll[i]
                })
    
    # Pandafy and store as csv
    df = pd.DataFrame.from_dict(yprdict) 
    df.to_csv(args.outcsv, index = False, header=True)

    # cleanup
    cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='whenet demo with yolo')
    parser.add_argument('--video', nargs='+', help='path to video file. use camera if no file is given')
    parser.add_argument('--snapshot', type=str, default='WHENet.h5', help='whenet snapshot path')
    parser.add_argument('--display', type=str, default='simple', help='display all euler angle (simple, full)')
    parser.add_argument('--score', type=float, default=0.3, help='yolo confidence score threshold')
    parser.add_argument('--iou', type=float, default=0.3, help='yolo iou threshold')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--output', default='test.avi', help='output video name')
    parser.add_argument('--outcsv', type=str, default='out.csv', help='output CSV file name')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)
