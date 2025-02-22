#import server.utils.config as config

#决定在预训练模型中采用三分类还是二分类
CLASSIFICATION_DIMENSION=3

POSITIVE=1
NEGATIVE=0
NEUTRAL=2
ABSTAIN=-1

train_dataset_path='datasets/restaurants-train.csv'
test_dataset_path='datasets/restaurants-test.csv'

#train_dataset_path='datasets/news11k_train.csv'
#test_dataset_path='datasets/news11k_test.csv'

aspect="Trump"