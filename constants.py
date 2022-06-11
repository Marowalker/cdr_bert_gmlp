from transformers import TFBertModel, BertTokenizer


MAX_SEN_LEN = 300
BATCH_SIZE = 16
DROPOUT = 0.1
LEARNING_RATE = 2e-5
NUM_EPOCH = 8
EMB_SIZE = 768
REBUILD = 1  # 1 to rebuild, 0 to load
REMAKE = 1  # 1 to train model again, 0 otherwise

relation = ['CID', 'NONE']

DATA = 'data/'
RAW_DATA = DATA + 'raw/'
PICKLE_DATA = DATA + 'pickle/'
BIOBERT = DATA + 'biobert_v1.0_pubmed_pmc'

TRAINED_MODELS = DATA + 'trained_models/'

encoder = TFBertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})

START_E1 = tokenizer.encode('<e1>')[1]
END_E1 = tokenizer.encode('</e1>')[1]
START_E2 = tokenizer.encode('<e2>')[1]
END_E2 = tokenizer.encode('</e2>')[1]

train_file = RAW_DATA + 'sentence_ref.train.txt'
dev_file = RAW_DATA + 'sentence_ref.dev.txt'
test_file = RAW_DATA + 'sentence_ref.test.txt'

train_token_pickle = PICKLE_DATA + 'train_x.pkl'
train_head_mask = PICKLE_DATA + 'train_x_head_mask.pkl'
train_e1_mask = PICKLE_DATA + 'train_x_e1_mask.pkl'
train_e2_mask = PICKLE_DATA + 'train_x_e2_mask.pkl'
train_labels = PICKLE_DATA + 'train_labels.pkl'

test_token_pickle = PICKLE_DATA + 'test_x.pkl'
test_head_mask = PICKLE_DATA + 'test_x_head_mask.pkl'
test_e1_mask = PICKLE_DATA + 'test_x_e1_mask.pkl'
test_e2_mask = PICKLE_DATA + 'test_x_e2_mask.pkl'
test_identities = PICKLE_DATA + 'test_identities.pkl'
