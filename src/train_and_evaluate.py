from src.masked_cross_entropy import *
from src.pre_data import *
from src.expressions_transfer import *
from src.models import *
import math
import torch
import torch.optim
import torch.nn.functional as f
import time

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    return target


def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_postfix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_postfix_expression(test) - compute_postfix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.ByteTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index.bool(), 0.0), masked_index


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


def train_tree_double(encoder_outputs, problem_output, all_nums_encoder_outputs, target, target_length,
                      output_lang, batch_size, padding_hidden, seq_mask, 
                      num_mask, num_pos, num_order_pad, nums_stack_batch, unk, 
                      encoder, numencoder, predict, generate, merge):
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

#    copy_num_len = [len(_) for _ in num_pos]
#    num_size = max(copy_num_len)
#    nums_encoder_outputs, masked_index = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
#                                                                        encoder.hidden_size)
#    all_nums_encoder_outputs = numencoder(nums_encoder_outputs, num_order_pad)
#    all_nums_encoder_outputs = all_nums_encoder_outputs.masked_fill_(masked_index.bool(), 0.0)
    
    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    # loss = loss_0 + loss_1
    return loss  # , loss_0.item(), loss_1.item()


def train_attn_double(encoder_outputs, decoder_hidden, target, target_length,
                      output_lang, batch_size, seq_mask, 
                      num_start, nums_stack_batch, unk,
                      decoder, beam_size, use_teacher_forcing):
    # Prepare input and output variables
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]] * batch_size)

    max_target_length = max(target_length)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()

    if random.random() < use_teacher_forcing:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            decoder_input = generate_decoder_input(
                target[t], decoder_output, nums_stack_batch, num_start, unk)
            target[t] = decoder_input
    else:
        beam_list = list()
        score = torch.zeros(batch_size)
        if USE_CUDA:
            score = score.cuda()
        beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs))
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            beam_len = len(beam_list)
            beam_scores = torch.zeros(batch_size, decoder.output_size * beam_len)
            all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2))
            all_outputs = torch.zeros(max_target_length, batch_size * beam_len, decoder.output_size)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
                all_outputs = all_outputs.cuda()

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                decoder_hidden = beam_list[b_idx].hidden

#                rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
#                                               num_start, copy_nums, generate_nums, english)
                if USE_CUDA:
#                    rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)

#                score = f.log_softmax(decoder_output, dim=1) + rule_mask
                score = f.log_softmax(decoder_output, dim=1)
                beam_score = beam_list[b_idx].score
                beam_score = beam_score.unsqueeze(1)
                repeat_dims = [1] * beam_score.dim()
                repeat_dims[1] = score.size(1)
                beam_score = beam_score.repeat(*repeat_dims)
                score += beam_score
                beam_scores[:, b_idx * decoder.output_size: (b_idx + 1) * decoder.output_size] = score
                all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                beam_list[b_idx].all_output[t] = decoder_output
                all_outputs[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                    beam_list[b_idx].all_output
            topv, topi = beam_scores.topk(beam_size, dim=1)
            beam_list = list()

            for k in range(beam_size):
                temp_topk = topi[:, k]
                temp_input = temp_topk % decoder.output_size
                temp_input = temp_input.data
                if USE_CUDA:
                    temp_input = temp_input.cpu()
                temp_beam_pos = temp_topk / decoder.output_size

                indices = torch.LongTensor(range(batch_size))
                if USE_CUDA:
                    indices = indices.cuda()
                indices += temp_beam_pos * batch_size

                temp_hidden = all_hidden.index_select(1, indices)
                temp_output = all_outputs.index_select(1, indices)

                beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_output))
        all_decoder_outputs = beam_list[0].all_output

        for t in range(max_target_length):
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], nums_stack_batch, num_start, unk)
    # Loss calculation and backpropagation

    if USE_CUDA:
        target = target.cuda()

    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target.transpose(0, 1).contiguous(),  # -> batch x seq
        target_length
    )
    
    return loss


def train_double(input1_batch, input2_batch, input_length, target1_batch, target1_length, target2_batch, target2_length, 
                 num_stack_batch, num_size_batch, generate_num1_ids, generate_num2_ids, copy_nums, 
                 encoder, numencoder, predict, generate, merge, decoder, 
                 encoder_optimizer, numencoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, decoder_optimizer, 
                 input_lang, output1_lang, output2_lang, num_pos_batch, num_order_batch, parse_graph_batch, 
                 beam_size=5, use_teacher_forcing=0.83, english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_num1_ids)
    for i in num_size_batch:
        d = i + len(generate_num1_ids)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)
    
    num_pos_pad = []
    max_num_pos_size = max(num_size_batch)
    for i in range(len(num_pos_batch)):
        temp = num_pos_batch[i] + [-1] * (max_num_pos_size-len(num_pos_batch[i]))
        num_pos_pad.append(temp)
    num_pos_pad = torch.LongTensor(num_pos_pad)

    num_order_pad = []
    max_num_order_size = max(num_size_batch)
    for i in range(len(num_order_batch)):
        temp = num_order_batch[i] + [0] * (max_num_order_size-len(num_order_batch[i]))
        num_order_pad.append(temp)
    num_order_pad = torch.LongTensor(num_order_pad)
    
    num_stack1_batch = copy.deepcopy(num_stack_batch)
    num_stack2_batch = copy.deepcopy(num_stack_batch)
    num_start2 = output2_lang.n_words - copy_nums - 2
    unk1 = output1_lang.word2index["UNK"]
    unk2 = output2_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input1_var = torch.LongTensor(input1_batch).transpose(0, 1)
    input2_var = torch.LongTensor(input2_batch).transpose(0, 1)
    target1 = torch.LongTensor(target1_batch).transpose(0, 1)
    target2 = torch.LongTensor(target2_batch).transpose(0, 1)
    parse_graph_pad = torch.LongTensor(parse_graph_batch)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    numencoder.train()
    predict.train()
    generate.train()
    merge.train()
    decoder.train()

    if USE_CUDA:
        input1_var = input1_var.cuda()
        input2_var = input2_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        num_pos_pad = num_pos_pad.cuda()
        num_order_pad = num_order_pad.cuda()
        parse_graph_pad = parse_graph_pad.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    numencoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # Run words through encoder
    
    encoder_outputs, encoder_hidden = encoder(input1_var, input2_var, input_length, parse_graph_pad)
    copy_num_len = [len(_) for _ in num_pos_batch]
    num_size = max(copy_num_len)
    num_encoder_outputs, masked_index = get_all_number_encoder_outputs(encoder_outputs, num_pos_batch, 
                                                                       batch_size, num_size, encoder.hidden_size)
    encoder_outputs, num_outputs, problem_output = numencoder(encoder_outputs, num_encoder_outputs, 
                                                              num_pos_pad, num_order_pad)
    num_outputs = num_outputs.masked_fill_(masked_index.bool(), 0.0)

    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    loss_0 = train_tree_double(encoder_outputs, problem_output, num_outputs, target1, target1_length,
                               output1_lang, batch_size, padding_hidden, seq_mask, 
                               num_mask, num_pos_batch, num_order_pad, num_stack1_batch, unk1, 
                               encoder, numencoder, predict, generate, merge)
    
    loss_1 = train_attn_double(encoder_outputs, decoder_hidden, target2, target2_length,
                               output2_lang, batch_size, seq_mask, 
                               num_start2, num_stack2_batch, unk2, 
                               decoder, beam_size, use_teacher_forcing)
    
    loss = loss_0 + loss_1
    loss.backward()
    
    encoder_optimizer.step()
    numencoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    decoder_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def evaluate_tree_double(encoder_outputs, problem_output, all_nums_encoder_outputs, 
                         output_lang, batch_size, padding_hidden, seq_mask, num_mask, 
                         max_length, num_pos, num_order_pad, 
                         encoder, numencoder, predict, generate, merge, beam_size):
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            topv, topi = out_score.topk(beam_size)

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0]


def evaluate_attn_double(encoder_outputs, decoder_hidden, 
                         output_lang, batch_size, seq_mask, max_length, 
                         decoder, beam_size):
    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]])  # SOS
    beam_list = list()
    score = 0
    beam_list.append(Beam(score, decoder_input, decoder_hidden, []))

    # Run through decoder
    for di in range(max_length):
        temp_list = list()
        beam_len = len(beam_list)
        for xb in beam_list:
            if int(xb.input_var[0]) == output_lang.word2index["EOS"]:
                temp_list.append(xb)
                beam_len -= 1
        if beam_len == 0:
            return beam_list[0]
        beam_scores = torch.zeros(decoder.output_size * beam_len)
        hidden_size_0 = decoder_hidden.size(0)
        hidden_size_2 = decoder_hidden.size(2)
        all_hidden = torch.zeros(beam_len, hidden_size_0, 1, hidden_size_2)
        if USE_CUDA:
            beam_scores = beam_scores.cuda()
            all_hidden = all_hidden.cuda()
        all_outputs = []
        current_idx = -1

        for b_idx in range(len(beam_list)):
            decoder_input = beam_list[b_idx].input_var
            if int(decoder_input[0]) == output_lang.word2index["EOS"]:
                continue
            current_idx += 1
            decoder_hidden = beam_list[b_idx].hidden

            # rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index,
            #                                1, num_start, copy_nums, generate_nums, english)
            if USE_CUDA:
                # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            # score = f.log_softmax(decoder_output, dim=1) + rule_mask.squeeze()
            score = f.log_softmax(decoder_output, dim=1)
            score += beam_list[b_idx].score
            beam_scores[current_idx * decoder.output_size: (current_idx + 1) * decoder.output_size] = score
            all_hidden[current_idx] = decoder_hidden
            all_outputs.append(beam_list[b_idx].all_output)
        topv, topi = beam_scores.topk(beam_size)

        for k in range(beam_size):
            word_n = int(topi[k])
            word_input = word_n % decoder.output_size
            temp_input = torch.LongTensor([word_input])
            indices = int(word_n / decoder.output_size)

            temp_hidden = all_hidden[indices]
            temp_output = all_outputs[indices]+[word_input]
            temp_list.append(Beam(float(topv[k]), temp_input, temp_hidden, temp_output))

        temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

        if len(temp_list) < beam_size:
            beam_list = temp_list
        else:
            beam_list = temp_list[:beam_size]
    return beam_list[0]


def evaluate_double(input1_batch, input2_batch, input_length, generate_num1_ids, generate_num2_ids, 
                    encoder, numencoder, predict, generate, merge, decoder, 
                    input_lang, output1_lang, output2_lang, num_pos_batch, num_order_batch, parse_graph_batch, 
                    beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    num_pos_pad = torch.LongTensor([num_pos_batch])
    num_order_pad = torch.LongTensor([num_order_batch])
    parse_graph_pad = torch.LongTensor(parse_graph_batch)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input1_var = torch.LongTensor(input1_batch).unsqueeze(1)
    input2_var = torch.LongTensor(input2_batch).unsqueeze(1)

    num_mask = torch.ByteTensor(1, len(num_pos_batch) + len(generate_num1_ids)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    numencoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()
    decoder.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input1_var = input1_var.cuda()
        input2_var = input2_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        num_pos_pad = num_pos_pad.cuda()
        num_order_pad = num_order_pad.cuda()
        parse_graph_pad = parse_graph_pad.cuda()
    # Run words through encoder

    encoder_outputs, encoder_hidden = encoder(input1_var, input2_var, [input_length], parse_graph_pad)
    num_size = len(num_pos_batch)
    num_encoder_outputs, masked_index = get_all_number_encoder_outputs(encoder_outputs, [num_pos_batch], batch_size, 
                                                                        num_size, encoder.hidden_size)
    encoder_outputs, num_outputs, problem_output = numencoder(encoder_outputs, num_encoder_outputs, 
                                                               num_pos_pad, num_order_pad)
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    
    tree_beam = evaluate_tree_double(encoder_outputs, problem_output, num_outputs, 
                                     output1_lang, batch_size, padding_hidden, seq_mask, num_mask, 
                                     max_length, num_pos_batch, num_order_pad, 
                                     encoder, numencoder, predict, generate, merge, beam_size)
    
    attn_beam = evaluate_attn_double(encoder_outputs, decoder_hidden, 
                                     output2_lang, batch_size, seq_mask, max_length, 
                                     decoder, beam_size)
    
    if tree_beam.score >= attn_beam.score:
        return "tree", tree_beam.out, tree_beam.score
    else:
        return "attn", attn_beam.all_output, attn_beam.score