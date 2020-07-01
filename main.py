# coding: utf-8
import os
import time
import torch.optim
import torch.nn as nn

from src.logger import *
from src.models import *
from src.train_and_evaluate import *
from src.expressions_transfer import *

batch_size = 64
embedding_size = 128
hidden_size = 512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2
hop_size = 2

from pyltp import Postagger,Parser
LTP_DATA_DIR="../ltp_data_v3.4.0"
pos_model_path = os.path.join(LTP_DATA_DIR, "pos.model")
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
postagger = Postagger()
postagger.load(pos_model_path)
parser = Parser()
parser.load(par_model_path)


def read_data_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def write_data_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def generate_train_test():
    data = load_raw_data("data/Math_23K.json")
    pairs, generate_nums, copy_nums = transfer_num(data)
    temp_pairs = []
    for p in pairs:
        if p[0] not in ["8883"]:
            temp_pairs.append((p[0], p[1], p[2], p[2], p[3], p[4]))
        else:
            temp_pairs.append((p[0], p[1], p[2], p[2], p[3], p[4]))

    pre_temp_pairs = []
    for p in temp_pairs:
        postags = postagger.postag(p[1])
        postags = ' '.join(postags).split(' ')
        arcs = parser.parse(p[1], postags)
        parse_tree = [arc.head-1 for arc in arcs]
        pre_temp_pairs.append((p[0], p[1], postags, parse_tree, 
                               from_infix_to_prefix(p[3]), from_infix_to_postfix(p[3]), p[4], p[5]))

    pairs = pre_temp_pairs

    fold_size = int(len(pairs) * 0.2)
    fold_pairs = []
    for split_fold in range(4):
        fold_start = fold_size * split_fold
        fold_end = fold_size * (split_fold + 1)
        fold_pairs.append(pairs[fold_start:fold_end])
    fold_pairs.append(pairs[(fold_size * 4):])

    for fold in range(5):
        pairs_tested = []
        pairs_trained = []
        for fold_t in range(5):
            if fold_t == fold:
                pairs_tested += fold_pairs[fold_t]
            else:
                pairs_trained += fold_pairs[fold_t]
        write_data_json(pairs_trained, "data/train_"+str(fold)+".json")
        write_data_json(pairs_tested, "data/test_"+str(fold)+".json")


def train(fold):
    data = load_raw_data("data/Math_23K.json")
    pairs, generate_nums, copy_nums = transfer_num(data)

    elogger = Logger("MultiMath_"+str(fold))
    pairs_trained = read_data_json("data/train_"+str(fold)+".json")
    pairs_tested = read_data_json("data/test_"+str(fold)+".json")

    best_acc_fold = []

    input1_lang, input2_lang, output1_lang, output2_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums, copy_nums)

    emb_vectors = word2vec(train_pairs, embedding_size, input1_lang)
    np.save("data/emb_"+str(fold)+".npy", emb_vectors)
    emb_vectors = np.load("data/emb_"+str(fold)+".npy")
    embed_model = nn.Embedding(input1_lang.n_words, embedding_size, padding_idx=0)
    embed_model.weight.data.copy_(torch.from_numpy(emb_vectors))

    # Initialize models
    encoder = EncoderSeq(input1_size=input1_lang.n_words, input2_size=input2_lang.n_words, 
                         embed_model=embed_model, embedding1_size=embedding_size, embedding2_size=embedding_size//4, 
                         hidden_size=hidden_size, n_layers=n_layers, hop_size=hop_size)
    numencoder = NumEncoder(node_dim=hidden_size, hop_size=hop_size)
    predict = Prediction(hidden_size=hidden_size, op_nums=output1_lang.n_words - copy_nums - 1 - len(generate_nums),
                         input_size=len(generate_nums))
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output1_lang.n_words - copy_nums - 1 - len(generate_nums),
                            embedding_size=embedding_size)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
    decoder = AttnDecoderRNN(hidden_size=hidden_size, embedding_size=embedding_size,
                             input_size=output2_lang.n_words, output_size=output2_lang.n_words, n_layers=n_layers)
    # the embedding layer is  only for generated number embeddings, operators, and paddings

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    numencoder_optimizer = torch.optim.Adam(numencoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
    numencoder_scheduler = torch.optim.lr_scheduler.StepLR(numencoder_optimizer, step_size=20, gamma=0.5)
    predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
    generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
    merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)
    decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=20, gamma=0.5)

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        numencoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()
        decoder.cuda()

    elogger.log(str(encoder))
    elogger.log(str(numencoder))
    elogger.log(str(predict))
    elogger.log(str(generate))
    elogger.log(str(merge))
    elogger.log(str(decoder))

    generate_num1_ids = []
    generate_num2_ids = []
    for num in generate_nums:
        generate_num1_ids.append(output1_lang.word2index[num])
        generate_num2_ids.append(output2_lang.word2index[num])

    for epoch in range(n_epochs):
        loss_total = 0
        id_batches, input1_batches, input2_batches, input_lengths, output1_batches, output1_lengths, output2_batches, output2_lengths, \
        nums_batches, num_stack_batches, num_pos_batches, num_order_batches, num_size_batches, parse_graph_batches = prepare_train_batch(train_pairs, batch_size)
        print("fold:", fold + 1)
        print("epoch:", epoch + 1)
        start = time.time()
        for idx in range(len(input_lengths)):
            loss = train_double(
                input1_batches[idx], input2_batches[idx], input_lengths[idx], output1_batches[idx], output1_lengths[idx], output2_batches[idx], output2_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx], generate_num1_ids, generate_num2_ids, copy_nums,
                encoder, numencoder, predict, generate, merge, decoder,
                encoder_optimizer, numencoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, decoder_optimizer,
                input1_lang, output1_lang, output2_lang, num_pos_batches[idx], num_order_batches[idx], parse_graph_batches[idx], 
                beam_size=5, use_teacher_forcing=0.83, english=False)
            loss_total += loss

        print("loss:", loss_total / len(input_lengths))
        print("training time", time_since(time.time() - start))
        print("--------------------------------")
        elogger.log("epoch: %d, loss: %.4f" % (epoch+1, loss_total/len(input_lengths)))

        if epoch % 10 == 0 or epoch > n_epochs - 5:
            value_ac = 0
            equation_ac = 0
            eval_total = 0
            result_list = []
            start = time.time()
            for test_batch in test_pairs:
                parse_graph = get_parse_graph_batch([test_batch[5]], [test_batch[4]])
                result_type, test_res, score = evaluate_double(test_batch[2], test_batch[3], test_batch[5], generate_num1_ids, generate_num2_ids,
                                                        encoder, numencoder, predict, generate, merge, decoder,
                                                        input1_lang, output1_lang, output2_lang, test_batch[11], test_batch[13], parse_graph, beam_size=beam_size)
                if result_type == "tree":
                    val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[6], output1_lang, test_batch[10], test_batch[12])
                    result = out_expression_list(test_res, output1_lang, test_batch[10])
                    result_list.append([test_batch[0], "tree", result, score])
                else:
                    if test_res[-1] == output2_lang.word2index["EOS"]:
                        test_res = test_res[:-1]
                    val_ac, equ_ac, _, _ = compute_postfix_tree_result(test_res, test_batch[8][:-1], output2_lang, test_batch[10], test_batch[12])
                    result = out_expression_list(test_res, output2_lang, test_batch[10])
                    result_list.append([test_batch[0], "attn", result, score])

                if val_ac:
                    value_ac += 1
                if equ_ac:
                    equation_ac += 1
                eval_total += 1
            print(equation_ac, value_ac, eval_total)
            print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
            print("testing time", time_since(time.time() - start))
            print("------------------------------------------------------")
            torch.save(encoder.state_dict(), "models_"+str(fold)+"/encoder")
            torch.save(numencoder.state_dict(), "models_"+str(fold)+"/numencoder")
            torch.save(predict.state_dict(), "models_"+str(fold)+"/predict")
            torch.save(generate.state_dict(), "models_"+str(fold)+"/generate")
            torch.save(merge.state_dict(), "models_"+str(fold)+"/merge")
            torch.save(decoder.state_dict(), "models_"+str(fold)+"/decoder")
            write_data_json(result_list, "results/result_"+str(fold)+".json")
            elogger.log("epoch: %d, test_equ_acc: %.4f, test_ans_acc: %.4f" \
                        % (epoch+1, float(equation_ac)/eval_total, float(value_ac)/eval_total))

            if epoch == n_epochs - 1:
                best_acc_fold.append((equation_ac, value_ac, eval_total))
        
        encoder_scheduler.step()
        numencoder_scheduler.step()
        predict_scheduler.step()
        generate_scheduler.step()
        merge_scheduler.step()
        decoder_scheduler.step()


def test():
    data = load_raw_data("data/Math_23K.json")
    pairs, generate_nums, copy_nums = transfer_num(data)
    
    fold = 0
    pairs_trained = read_data_json("data/train_"+str(fold)+".json")
    pairs_tested = read_data_json("data/test_"+str(fold)+".json")
    
    input1_lang, input2_lang, output1_lang, output2_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums, copy_nums)
    
    emb_vectors = np.load("data/emb_"+str(fold)+".npy")
    embed_model = nn.Embedding(input1_lang.n_words, embedding_size)
    embed_model.weight.data.copy_(torch.from_numpy(emb_vectors))
    
    # Initialize models
    encoder = EncoderSeq(input1_size=input1_lang.n_words, input2_size=input2_lang.n_words, 
                         embed_model=embed_model, embedding1_size=embedding_size, embedding2_size=embedding_size//4, 
                         hidden_size=hidden_size, n_layers=n_layers, hop_size=hop_size)
    numencoder = NumEncoder(node_dim=hidden_size, hop_size=hop_size)
    predict = Prediction(hidden_size=hidden_size, op_nums=output1_lang.n_words - copy_nums - 1 - len(generate_nums),
                         input_size=len(generate_nums))
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output1_lang.n_words - copy_nums - 1 - len(generate_nums),
                            embedding_size=embedding_size)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
    decoder = AttnDecoderRNN(hidden_size=hidden_size, embedding_size=embedding_size,
                             input_size=output2_lang.n_words, output_size=output2_lang.n_words, n_layers=n_layers)
    
    encoder.load_state_dict(torch.load("models_"+str(fold)+"/encoder", map_location="cpu"))
    numencoder.load_state_dict(torch.load("models_"+str(fold)+"/numencoder", map_location="cpu"))
    predict.load_state_dict(torch.load("models_"+str(fold)+"/predict", map_location="cpu"))
    generate.load_state_dict(torch.load("models_"+str(fold)+"/generate", map_location="cpu"))
    merge.load_state_dict(torch.load("models_"+str(fold)+"/merge", map_location="cpu"))
    decoder.load_state_dict(torch.load("models_"+str(fold)+"/decoder", map_location="cpu"))
    
    if USE_CUDA:
        encoder.cuda()
        numencoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()
        decoder.cuda()
    
    generate_num1_ids = []
    generate_num2_ids = []
    for num in generate_nums:
        generate_num1_ids.append(output1_lang.word2index[num])
        generate_num2_ids.append(output2_lang.word2index[num])
    
    pair = pairs_tested[211][:]
    pair[1] = ['快车', '每', '小时', '行驶', 'NUM', '千米', '，', '慢车', '每', '小时', '行驶', 'NUM', '千米', '，', 
               '两车', '相向', '而', '行', '，', '经过', 'NUM', '小时', '相遇', '，', '相遇', '时', '快车', '比', '慢车', '多行', '多少', '千米', '？']
    postags = postagger.postag(pair[1])
    postags = ' '.join(postags).split(' ')
    arcs = parser.parse(pair[1], postags)
    parse_tree = [arc.head-1 for arc in arcs]
    pair[2] = postags
    pair[3] = parse_tree
    pair[4] = ['*', '-', 'N0', 'N1', 'N2']
    pair[5] = ['N0', 'N1', '-', 'N2', '*']
    pair[6] = ['85', '58', '5']
    pair[7] = [4, 11, 20]
#    
#    pair = pairs_tested[211][:]
#    pair[1] = ['慢车', '每', '小时', '行驶', 'NUM', '千米', '，', '快车', '每', '小时', '行驶', 'NUM', '千米', '，', 
#               '两车', '相向', '而', '行', '，', '经过', 'NUM', '小时', '相遇', '，', '相遇', '时', '慢车', '比', '快车', '少行', '多少', '千米', '？']
#    postags = postagger.postag(pair[1])
#    postags = ' '.join(postags).split(' ')
#    arcs = parser.parse(pair[1], postags)
#    parse_tree = [arc.head-1 for arc in arcs]
#    pair[2] = postags
#    pair[3] = parse_tree
    
#    pair = pairs_tested[45][:]
#    pair[1] = ['妈妈', '有', 'NUM', '米', '蓝', '带子', '，', 
#               'NUM', '米', '红带子', '．', '蓝', '带子', '是', '红带子', '的', '几分', '之' '几', '？']
#    pair[2] = ['/', 'N0', 'N1']
#    pair[3] = ['N0', 'N1', '/']
#    pair[4] = ['3', '12']
#    pair[5] = [2, 7]

#    pair = pairs_tested[45][:]
#    pair[1] = ['妈妈', '有', 'NUM', '米', '蓝', '带子', '，', 
#               'NUM', '米', '红带子', '．', '蓝', '带子', '的', '长', '是', '红带子', '的', '几倍', '？']
#    pair[2] = ['/', 'N0', 'N1']
#    pair[3] = ['N0', 'N1', '/']
#    pair[4] = ['12', '3']
#    pair[5] = [2, 7]
    
    test_pairs = []
    num_stack = []
    for word in pair[4]:
        temp_num = []
        flag_not = True
        if word not in output1_lang.index2word:
            flag_not = False
            for i, j in enumerate(pair[6]):
                if j == word:
                    temp_num.append(i)
    
        if not flag_not and len(temp_num) != 0:
            num_stack.append(temp_num)
        if not flag_not and len(temp_num) == 0:
            num_stack.append([_ for _ in range(len(pair[6]))])
    
    num_stack.reverse()
    input1_cell = indexes_from_sentence(input1_lang, pair[1])
    texts_cell = texts_from_sentence(input1_lang, pair[1])
    input2_cell = indexes_from_sentence(input2_lang, pair[2])
    output1_cell = indexes_from_sentence(output1_lang, pair[4], True)
    output2_cell = indexes_from_sentence(output2_lang, pair[5], False)
    num_list = num_list_processed(pair[6])
    num_order = num_order_processed(num_list)
    test_pairs.append((pair[0], texts_cell, input1_cell, input2_cell, pair[3], len(input1_cell), 
                       output1_cell, len(output1_cell), output2_cell, len(output2_cell), 
                       pair[6], pair[7], num_stack, num_order))
    
    for test_batch in test_pairs:
        parse_graph = get_parse_graph_batch([test_batch[5]], [test_batch[4]])
        result_type, test_res, score = evaluate_double(test_batch[2], test_batch[3], test_batch[5], generate_num1_ids, generate_num2_ids,
                                                encoder, numencoder, predict, generate, merge, decoder,
                                                input1_lang, output1_lang, output2_lang, test_batch[11], test_batch[13], parse_graph, beam_size=beam_size)
        if result_type == "tree":
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[6], output1_lang, test_batch[10], test_batch[12])
            result = out_expression_list(test_res, output1_lang, test_batch[10])
        else:
            if test_res[-1] == output2_lang.word2index["EOS"]:
                test_res = test_res[:-1]
            val_ac, equ_ac, _, _ = compute_postfix_tree_result(test_res, test_batch[8][:-1], output2_lang, test_batch[10], test_batch[12])
            result = out_expression_list(test_res, output2_lang, test_batch[10])
    print(result)

if __name__ == '__main__':
#    train(0)
#    train(1)
#    train(2)
#    train(3)
#    train(4)
#    test()
    print('test')
