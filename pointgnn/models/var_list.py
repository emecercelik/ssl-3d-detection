''' This file contains the variable lists for the Point-GNN backbone, detection head, and scene flow head'''

backbone = ['layer4/extract_vertex_features/fully_connected_1/biases', 'layer2/combined_features/fully_connected_1/weights', 'layer3/extract_vertex_features/fully_connected_1/biases', 'layer2/fully_connected/biases', 'layer2/fully_connected_1/biases', 'layer2/extract_vertex_features/fully_connected_1/weights', 'layer3/fully_connected_1/biases', 'layer4/fully_connected/biases', 'layer1/extract_vertex_features/fully_connected_2/weights', 'layer4/combined_features/fully_connected/weights', 'layer3/fully_connected_1/weights', 'layer4/combined_features/fully_connected_1/weights', 'layer2/combined_features/fully_connected/biases', 'layer1/combined_features/fully_connected_1/biases', 'layer1/extract_vertex_features/fully_connected/weights', 'layer3/combined_features/fully_connected/weights', 'layer3/combined_features/fully_connected/biases', 'layer3/extract_vertex_features/fully_connected_1/weights', 'layer1/extract_vertex_features/fully_connected_3/biases', 'layer4/combined_features/fully_connected_1/biases', 'layer4/extract_vertex_features/fully_connected/biases', 'layer3/extract_vertex_features/fully_connected/biases', 'layer3/fully_connected/biases', 'layer3/combined_features/fully_connected_1/weights', 'layer4/fully_connected_1/weights', 'layer1/combined_features/fully_connected/biases', 'layer2/combined_features/fully_connected_1/biases', 'layer1/extract_vertex_features/fully_connected/biases', 'layer3/extract_vertex_features/fully_connected/weights', 'layer2/fully_connected/weights', 'layer4/combined_features/fully_connected/biases', 'layer1/extract_vertex_features/fully_connected_2/biases', 'layer1/combined_features/fully_connected_1/weights', 'layer4/extract_vertex_features/fully_connected/weights', 'layer4/fully_connected/weights', 'layer2/extract_vertex_features/fully_connected_1/biases', 'layer2/extract_vertex_features/fully_connected/weights', 'layer2/combined_features/fully_connected/weights', 'layer1/combined_features/fully_connected/weights', 'layer1/extract_vertex_features/fully_connected_3/weights', 'layer2/fully_connected_1/weights', 'layer3/combined_features/fully_connected_1/biases', 'layer1/extract_vertex_features/fully_connected_1/biases', 'layer4/fully_connected_1/biases', 'layer4/extract_vertex_features/fully_connected_1/weights', 'layer2/extract_vertex_features/fully_connected/biases', 'layer3/fully_connected/weights', 'layer1/extract_vertex_features/fully_connected_1/weights']



detection_head = ['output/predictor/loc/cls_2/fully_connected/biases', 'output/predictor/loc/cls_2/fully_connected_2/biases', 'output/predictor/loc/cls_3/fully_connected/biases', 'output/predictor/cls/fully_connected/biases', 'output/predictor/loc/cls_0/fully_connected_1/weights', 'output/predictor/cls/fully_connected_1/biases', 'output/predictor/loc/cls_2/fully_connected_2/weights', 'output/predictor/loc/cls_2/fully_connected/weights', 'output/predictor/loc/cls_2/fully_connected_1/weights', 'output/predictor/loc/cls_0/fully_connected/weights', 'output/predictor/cls/fully_connected/weights', 'output/predictor/loc/cls_1/fully_connected_1/weights', 'output/predictor/loc/cls_3/fully_connected_2/weights', 'output/predictor/loc/cls_2/fully_connected_1/biases', 'output/predictor/loc/cls_1/fully_connected_1/biases', 'output/predictor/cls/fully_connected_1/weights', 'output/predictor/loc/cls_0/fully_connected_1/biases', 'output/predictor/loc/cls_1/fully_connected/biases', 'output/predictor/loc/cls_3/fully_connected/weights', 'output/predictor/loc/cls_3/fully_connected_2/biases', 'output/predictor/loc/cls_0/fully_connected/biases', 'output/predictor/loc/cls_1/fully_connected_2/biases', 'output/predictor/loc/cls_3/fully_connected_1/biases', 'output/predictor/loc/cls_0/fully_connected_2/biases', 'output/predictor/loc/cls_1/fully_connected_2/weights', 'output/predictor/loc/cls_1/fully_connected/weights', 'output/predictor/loc/cls_0/fully_connected_2/weights', 'output/predictor/loc/cls_3/fully_connected_1/weights']


scene_flow_head = ['flownet3d/flow_embedding/conv_diff_0/biases', 'flownet3d/layer4/conv1/bn/gamma', 'flownet3d/flow_embedding/conv_diff_2/bn/moving_mean', 'flownet3d/up_sa_layer1/post-conv0/bn/gamma', 'flownet3d/layer3/conv1/bn/moving_mean', 'flownet3d/layer3/conv1/biases', 'flownet3d/sa1/layer2/conv2/bn/beta', 'flownet3d/layer4/conv1/biases', 'flownet3d/sa1/layer2/conv2/weights', 'flownet3d/up_sa_layer2/conv2/weights', 'flownet3d/up_sa_layer2/conv0/bn/beta', 'flownet3d/layer4/conv0/bn/beta', 'flownet3d/sa1/layer2/conv1/bn/beta', 'flownet3d/sa1/layer2/conv1/weights', 'flownet3d/up_sa_layer2/conv0/bn/moving_variance', 'flownet3d/layer3/conv1/bn/beta', 'flownet3d/up_sa_layer2/conv0/bn/moving_mean', 'flownet3d/layer4/conv1/bn/beta', 'flownet3d/up_sa_layer1/post-conv1/bn/moving_variance', 'flownet3d/up_sa_layer1/post-conv0/bn/beta', 'flownet3d/layer3/conv0/bn/moving_mean', 'flownet3d/up_sa_layer1/post-conv1/weights', 'flownet3d/fa_layer4/conv_1/bn/beta', 'flownet3d/layer3/conv2/weights', 'flownet3d/layer3/conv0/bn/moving_variance', 'flownet3d/up_sa_layer2/conv2/bn/gamma', 'flownet3d/up_sa_layer1/post-conv0/weights', 'flownet3d/flow_embedding/conv_diff_2/weights', 'flownet3d/layer3/conv2/bn/beta', 'flownet3d/sa1/layer2/conv1/biases', 'flownet3d/fa_layer4/conv_1/bn/moving_variance', 'flownet3d/fa_layer4/conv_1/bn/gamma', 'flownet3d/layer4/conv1/bn/moving_mean', 'flownet3d/fa_layer4/conv_1/weights', 'flownet3d/fa_layer4/conv_0/biases', 'flownet3d/flow_embedding/conv_diff_1/bn/gamma', 'flownet3d/fa_layer4/conv_1/biases', 'flownet3d/layer3/conv1/weights', 'flownet3d/layer4/conv2/bn/beta', 'flownet3d/flow_embedding/conv_diff_2/biases', 'flownet3d/up_sa_layer1/post-conv1/biases', 'flownet3d/layer3/conv0/biases', 'flownet3d/flow_embedding/conv_diff_2/bn/beta', 'flownet3d/layer4/conv0/biases', 'flownet3d/fc1/biases', 'flownet3d/up_sa_layer2/conv1/bn/beta', 'flownet3d/layer4/conv2/biases', 'flownet3d/layer4/conv2/bn/moving_mean', 'flownet3d/sa1/layer2/conv1/bn/moving_variance', 'flownet3d/flow_embedding/conv_diff_0/weights', 'flownet3d/up_sa_layer2/conv1/weights', 'flownet3d/up_sa_layer2/conv2/bn/moving_variance', 'flownet3d/up_sa_layer1/post-conv1/bn/gamma', 'flownet3d/flow_embedding/conv_diff_1/bn/moving_mean', 'flownet3d/layer3/conv2/bn/gamma', 'flownet3d/fa_layer4/conv_0/bn/moving_mean', 'flownet3d/layer4/conv2/weights', 'flownet3d/flow_embedding/conv_diff_1/biases', 'flownet3d/fc1/bn/beta', 'flownet3d/layer3/conv2/bn/moving_variance', 'flownet3d/sa1/layer2/conv0/bn/moving_variance', 'flownet3d/fc1/bn/gamma', 'flownet3d/sa1/layer2/conv2/bn/moving_mean', 'flownet3d/up_sa_layer2/conv0/bn/gamma', 'flownet3d/flow_embedding/conv_diff_0/bn/beta', 'flownet3d/fc1/weights', 'flownet3d/up_sa_layer2/post-conv0/bn/gamma', 'flownet3d/fc1/bn/moving_variance', 'flownet3d/up_sa_layer1/post-conv0/biases', 'flownet3d/sa1/layer2/conv2/bn/gamma', 'flownet3d/sa1/layer2/conv0/weights', 'flownet3d/layer4/conv1/bn/moving_variance', 'flownet3d/up_sa_layer2/post-conv0/weights', 'flownet3d/layer4/conv1/weights', 'flownet3d/up_sa_layer2/conv1/bn/moving_variance', 'flownet3d/fa_layer4/conv_0/bn/gamma', 'flownet3d/up_sa_layer2/conv2/bn/moving_mean', 'flownet3d/up_sa_layer2/conv2/biases', 'flownet3d/flow_embedding/conv_diff_2/bn/gamma', 'flownet3d/flow_embedding/conv_diff_0/bn/gamma', 'flownet3d/layer4/conv2/bn/moving_variance', 'flownet3d/layer4/conv0/bn/gamma', 'flownet3d/up_sa_layer2/post-conv0/bn/beta', 'flownet3d/up_sa_layer2/post-conv0/bn/moving_mean', 'flownet3d/layer3/conv1/bn/moving_variance', 'flownet3d/layer3/conv0/bn/beta', 'flownet3d/sa1/layer2/conv0/bn/moving_mean', 'flownet3d/layer3/conv1/bn/gamma', 'flownet3d/up_sa_layer2/post-conv0/bn/moving_variance', 'flownet3d/layer3/conv0/bn/gamma', 'flownet3d/fa_layer4/conv_0/bn/moving_variance', 'flownet3d/fa_layer4/conv_1/bn/moving_mean', 'flownet3d/up_sa_layer2/conv0/biases', 'flownet3d/up_sa_layer2/conv1/bn/moving_mean', 'flownet3d/sa1/layer2/conv0/bn/gamma', 'flownet3d/fa_layer4/conv_0/weights', 'flownet3d/sa1/layer2/conv0/biases', 'flownet3d/fc2/weights', 'flownet3d/flow_embedding/conv_diff_1/bn/moving_variance', 'flownet3d/sa1/layer2/conv2/bn/moving_variance', 'flownet3d/up_sa_layer1/post-conv1/bn/beta', 'flownet3d/flow_embedding/conv_diff_0/bn/moving_mean', 'flownet3d/up_sa_layer2/conv0/weights', 'flownet3d/up_sa_layer2/conv1/bn/gamma', 'flownet3d/layer3/conv2/biases', 'flownet3d/flow_embedding/conv_diff_2/bn/moving_variance', 'flownet3d/flow_embedding/conv_diff_0/bn/moving_variance', 'flownet3d/layer4/conv0/weights', 'flownet3d/layer4/conv2/bn/gamma', 'flownet3d/up_sa_layer1/post-conv0/bn/moving_mean', 'flownet3d/fc2/biases', 'flownet3d/up_sa_layer2/conv1/biases', 'flownet3d/layer4/conv0/bn/moving_mean', 'flownet3d/layer3/conv0/weights', 'flownet3d/sa1/layer2/conv1/bn/gamma', 'flownet3d/sa1/layer2/conv2/biases', 'flownet3d/up_sa_layer2/conv2/bn/beta', 'flownet3d/sa1/layer2/conv0/bn/beta', 'flownet3d/up_sa_layer1/post-conv1/bn/moving_mean', 'flownet3d/flow_embedding/conv_diff_1/weights', 'flownet3d/layer4/conv0/bn/moving_variance', 'flownet3d/up_sa_layer1/post-conv0/bn/moving_variance', 'flownet3d/sa1/layer2/conv1/bn/moving_mean', 'flownet3d/fc1/bn/moving_mean', 'flownet3d/layer3/conv2/bn/moving_mean', 'flownet3d/flow_embedding/conv_diff_1/bn/beta', 'flownet3d/fa_layer4/conv_0/bn/beta', 'flownet3d/up_sa_layer2/post-conv0/biases']

