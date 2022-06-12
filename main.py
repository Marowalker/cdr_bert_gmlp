from preprocessing import *
from bert_model import BertgMLPModel
from constants import REBUILD
from utils import *
from evaluate.bc5 import evaluate_bc5


def main():
    train_in = [train_file, dev_file]
    train_out = [train_token_pickle, train_head_mask, train_e1_mask, train_e2_mask, train_labels]

    test_in = [test_file]
    test_out = [test_token_pickle, test_head_mask, test_e1_mask, test_e2_mask, test_identities]

    if REBUILD == 1:
        print('Build data...')
        train_x, train_x_head, train_x_e1, train_x_e2, train_y = make_pickle(train_in, train_out, case='train')
        test_x, test_x_head, test_x_e1, test_x_e2, identities = make_pickle(test_in, test_out, case='test')
    else:
        print('Load data...')
        train_x, train_x_head, train_x_e1, train_x_e2, train_y = load_pickle(train_out, case='train')
        test_x, test_x_head, test_x_e1, test_x_e2, identities = load_pickle(test_out, case='test')

    print('Data obtained with size:', len(train_y))

    model = BertgMLPModel(encoder, depth=10)
    model.build(train_x, train_x_head, train_x_e1, train_x_e2, train_y)
    y_pred = model.predict(test_x, test_x_head, test_x_e1, test_x_e2)

    answer = {}

    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            if identities[i][0] not in answer:
                answer[identities[i][0]] = []

            if identities[i][1] not in answer[identities[i][0]]:
                answer[identities[i][0]].append(identities[i][1])

    print(
            'result: abstract: ', evaluate_bc5(answer)
    )


if __name__ == '__main__':
    main()
