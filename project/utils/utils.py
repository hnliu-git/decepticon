import torch
import numpy as np


def default_collate_fn(batch, tokenizer):
    """"""
    articles = []
    questions = []
    answers = []
    distractors = []

    for item in batch:
        articles.append(item["article"])
        questions.append(item["question"])
        answers.append(item["answer"])
        distractors.append(tokenizer.additional_special_tokens[-1].join(item["distractors"]))

    return {
        "articles": tokenizer(articles, padding=True, truncation=True, max_length=500, return_tensors="pt"),
        "questions": tokenizer(questions, padding=True, return_tensors="pt"),
        "answers": tokenizer(answers, padding=True, return_tensors="pt"),
        "distractors": tokenizer(distractors, padding=True, return_tensors="pt"),
    }


def t5_collate_fn(batch, tokenizer):
    """"""
    context = []
    questions = []
    for item in batch:
        context.append(" ".join(["<ANS>", item["answer"], "<CON>", item["article"]]))
        questions.append(item["question"])
    context = tokenizer(text=context,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        pad_to_max_length=True,
                        max_length=512)
    questions = tokenizer(questions,
                          padding=True,
                          truncation=True,
                          return_tensors="pt",
                          pad_to_max_length=True,
                          max_length=512)

    context['input_ids'] = torch.squeeze(context['input_ids'])
    context['attention_mask'] = torch.squeeze(context['attention_mask'])
    questions['input_ids'] = torch.squeeze(questions['input_ids'])
    questions['attention_mask'] = torch.squeeze(questions['attention_mask'])

    return context, questions


def t5_dis_collate_fn(batch, tokenizer):
    """"""
    context = []
    distractor = []
    for item in batch:
        context.append(
            " ".join(["<ANS>", item["answer"], "<QUE>", item["question"], "<CON>", item["article"]]))
        indx = np.random.randint(low=0, high=len(item["distractors"]), size=1)[0]
        distractor.append(item["distractors"][indx])

    context = tokenizer(text=context,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        pad_to_max_length=True,
                        max_length=512)

    distractor = tokenizer(distractor,
                           padding=True,
                           truncation=True,
                           return_tensors="pt",
                           pad_to_max_length=True,
                           max_length=512)

    context['input_ids'] = torch.squeeze(context['input_ids'])
    context['attention_mask'] = torch.squeeze(context['attention_mask'])
    distractor['input_ids'] = torch.squeeze(distractor['input_ids'])
    distractor['attention_mask'] = torch.squeeze(distractor['attention_mask'])

    return context, distractor


def transformer_collate_fn(batch, tokenizer):
    """"""
    con_token, que_token, ans_token, dis_token = tokenizer.additional_special_tokens

    inputs = []
    targets = []

    for item in batch:
        inputs.append(" ".join([con_token, item["article"], ans_token, item["answer"]]))
        targets.append(" ".join([que_token, item["question"], dis_token, dis_token.join(item["distractors"])]))

    return {
        "inputs": tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors="pt"),
        "targets": tokenizer(targets, padding=True, truncation=True, return_tensors="pt"),
    }


def rnn_batch_fn(batch):
    """
    Description: from batch to x, y
    """
    art = batch['articles']['input_ids']
    que = batch['questions']['input_ids']
    ans = batch['answers']['input_ids']
    x, y = torch.cat([ans, art], dim=1).long(), que.long()
    return x, y


def rnn_dis_batch_fn(batch):
    """"""
    art = batch['articles']['input_ids']
    que = batch['questions']['input_ids']
    ans = batch['answers']['input_ids']
    dis = batch["distractors"]['input_ids']
    x, y = torch.cat([que, ans, art], dim=1).long(), dis.long()
    return x, y


def display_result_as_string(tokenizer, ans, output, tgt):
    """
    Args:
        vocab dictionary of [index, word]
        ans (bsz, seq_len) Tensor
        tgt (bsz, seq_len) Tensor
        output (bsz, seq_len, vocab_size) OR (bsz, seq_len) Tensor
    """
    ans = ans[0, :].long().numpy()
    tgt = tgt[0, :].long().numpy()
    if len(output.shape) == 3:
        output = output[0, :, :].numpy()
        output = np.argmax(output, axis=1)
    else:
        output = output[0, :].long().numpy()
    ans_str = ' '.join(tokenizer.convert_ids_to_tokens(ans, True))
    tgt_str = ' '.join(tokenizer.convert_ids_to_tokens(tgt, True))
    out_str = ' '.join(tokenizer.convert_ids_to_tokens(output, True))
    print("\n============================")
    print("ANS:", ans_str)
    print("TGT:", tgt_str)
    print("OUT:", out_str)


