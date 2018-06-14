input_file = "../process.csv.bz2"
# SEP = "\t"
SEP = ","
w2vpath = '../vectors_big.txt'
# w2vpath = '../baike.128.no_truncate.glove.txt'
embedding_matrix_path = './matrix_big.npy'

word_index_path = "worddict.pkl"
TRAIN_HDF5 = "train_hdf5_small.h5"

MAX_TEXT_LENGTH = 500
MAX_FEATURES = 200000
embedding_dims = 200

fit_batch_size = 32
fit_epoch = 40

class_num = 202
law_class_num = 183
time_class_num = 9