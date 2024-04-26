import os
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

def process_video(ori_data_path, video, action_name, save_dir):
    resize_height = 128
    resize_width = 171

    # Initialize a VideoCapture object to read video data into a numpy array
    video_filename = video.split('.')[0]
    if not os.path.exists(os.path.join(save_dir, video_filename)):
        os.mkdir(os.path.join(save_dir, video_filename))

    capture = cv2.VideoCapture(os.path.join(ori_data_path, action_name, video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Make sure splited video has at least 16 frames
    EXTRACT_FREQUENCY = 4
    if frame_count // EXTRACT_FREQUENCY <= 16:
        EXTRACT_FREQUENCY -= 1
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1

    count = 0
    i = 0
    retaining = True

    while (count < frame_count and retaining):
        retaining, frame = capture.read()
        if frame is None:
            continue

        if count % EXTRACT_FREQUENCY == 0:
            if (frame_height != resize_height) or (frame_width != resize_width):
                frame = cv2.resize(frame, (resize_width, resize_height))
            cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
            i +=1
        count +=1

    #Release the VideoCapture once it is no longer needed
    capture.release()


def preprocess(ori_data_path, output_data_path):
    #查看是否存在输入文件地址，如果没有则创建，同时创建， train,  val,  test文件夹
    if not os.path.exists(output_data_path):
        os.mkdir(output_data_path)
        os.mkdir(os.path.join(output_data_path, 'train'))
        os.mkdir(os.path.join(output_data_path, 'val'))
        os.mkdir(os.path.join(output_data_path, 'test'))

    #获取原始文件下的所有的类别文件的路径
    for file in os.listdir(ori_data_path):
        file_path = os.path.join(ori_data_path, file)
        #获取每个类别文件下的视频类别名
        video_file = [name for name in os.listdir(file_path)]
        #划分类别名下的所有视频元素
        train_and_valid, test = train_test_split(video_file, test_size=0.2, random_state=42)
        train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

        #生成对应类别名视频名的文件路径
        train_dir = os.path.join(output_data_path, 'train', file)
        val_dir = os.path.join(output_data_path, 'val', file)
        test_dir = os.path.join(output_data_path, 'test', file)
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)


        #分别对train， val， test数据做操作
        for video in train:
            process_video(ori_data_path, video, file, train_dir)
        for video in val:
            process_video(ori_data_path, video, file, val_dir)
        for video in test:
            process_video(ori_data_path, video, file, test_dir)
        print('{}划分完成'.format(file))
    print('所有数据划分完成')


def label_text_write(ori_dat_path, out_label_path):
    folder = ori_dat_path
    fnames, labels = [], []
    for label in sorted(os.listdir(folder)):
        for fname in os.listdir(os.path.join(folder, label)):
            fnames.append(os.path.join(folder, label, fname))
            labels.append(label)

    label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
    if not os.path.exists(out_label_path + '/labels.txt'):
        with open(out_label_path + '/labels.txt', 'w') as f:
            for id, label in enumerate(sorted(label2index)):
                f.writelines(str(id+1) + ' ' + label + '\n')



if __name__ == "__main__":
    ori_data_path = 'data/UCF-101'
    out_label_path = 'data'
    output_data_path = 'data/ucf'

    #生成标签文档
    # label_text_write(ori_data_path, out_label_path)

    #划分数据集，生成对应的图片数据集
    preprocess(ori_data_path, output_data_path)




