function [scores, face_label] = face_detection(img)
addpath('..');
base_path = '../../models/face_detect_caffenet/';
prototxt_dir = [base_path, 'deploy.prototxt'];
model_dir = [base_path, 'face_detect.caffemodel'];
d = load('../+caffe/imagenet/ilsvrc_2012_mean.mat');
%% mean file
IMAGE_MEAN = d.mean_data;
caffe.reset_all();
%% image size
img_size_info = [size(img, 1), size(img, 2)];
%get_test_prototxt(prototxt_dir, img_size_info);
caffe.set_mode_gpu();
gpu_id = 0;
caffe.set_device(gpu_id);

phase = 'test';
net = caffe.Net(prototxt_dir, model_dir, phase);
%% prepare image
img = imresize(img,img_size_info, 'bilinear');
IMAGE_MEAN = imresize(IMAGE_MEAN,img_size_info, 'bilinear');
img = single(img(:,:,[3 2 1])) - IMAGE_MEAN;
img = permute(img, [2, 1, 3]);
input_data = {img};

tic;
f = net.forward(input_data);
toc;

scores = f{1};
scores = mean(scores, 2);
[~, face_label] = max(scores);
face_label = face_label - 1;
caffe.reset_all();
