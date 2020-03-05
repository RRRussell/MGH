import os

base_dir =  "/home/dwu/AneurysmDetection/dzh/FP_Reduction/test/"
output_dir = base_dir+"FP_DATA/"
annatation_file = base_dir + 'CSVFILES/annotations.csv'
candidate_file = base_dir +  'CSVFILES/candidates.csv'
plot_output_path = output_dir + 'cubic_npy/'
if not os.path.exists(plot_output_path):
    os.mkdir(plot_output_path)
normalazation_output_path = output_dir + 'cubic_normalization_npy/'
if not os.path.exists(normalazation_output_path):
    os.mkdir(normalazation_output_path)
test_path = output_dir + 'cubic_npy/'
if not os.path.exists(test_path):
    os.mkdir(test_path)
###  training and test configuration #####
batch_size = 32
learning_rate = 0.0001
keep_prob = 1