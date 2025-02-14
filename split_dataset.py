import json

number_split = 3

# with open('dataset/hotpot_train_v1.json', 'r') as file:
#     train = json.load(file)

# with open('dataset/hotpot_train_v1.json', 'r') as file:
#     train = json.load(file)

# with open('dataset/hotpot_train_v1.json', 'r') as file:
#     train = json.load(file)

# print(len(train))
# for number in range(number_split):
#     with open(f'dataset/hotpot_train_v1_{number}.json', 'w') as f:
#         json.dump(train[int(len(train)*number/number_split):int(len(train)*(number+1)/number_split)], f, indent=4)

# for number in range(number_split):
#     with open(f'dataset/hotpot_train_v1_{number}.json', 'w') as f:
#         json.dump(train[int(len(train)*number/number_split):int(len(train)*(number+1)/number_split)], f, indent=4)

# for number in range(number_split):
#     with open(f'dataset/hotpot_train_v1_{number}.json', 'w') as f:
#         json.dump(train[int(len(train)*number/number_split):int(len(train)*(number+1)/number_split)], f, indent=4)


# with open('dataset/hotpot_train_v1_0.json', 'r') as file:
#     train_0 = json.load(file)

# with open('dataset/hotpot_train_v1_1.json', 'r') as file:
#     train_1 = json.load(file)

# with open('dataset/hotpot_train_v1_2.json', 'r') as file:
#     train_2 = json.load(file)

# print(len(train_0))
# print(len(train_1))
# print(len(train_2))

# for number in range(number_split):
#     print(
#         f"python main.py --mode prepro --data_file dataset/hotpot_train_v1_{number}.json --word_emb_file prepro_result/train/{number}/word_emb_file_{number}.json --char_emb_file prepro_result/train/{number}/char_emb_file_{number}.json --word2idx_file prepro_result/train/{number}/word2idx_file_{number}.json --char2idx_file prepro_result/train/{number}/char2idx_file_{number}.json --idx2word_file prepro_result/train/{number}/idx2word_file_{number}.json --idx2char_file prepro_result/train/{number}/idx2char_file_{number}.json --train_eval_file prepro_result/train/{number}/char2idx_file_{number}.json --para_limit 5000 --train_eval_file prepro_result/train/{number}/train_eval_{number}.json --data_split train"
#     )

for number in range(number_split):
    print(
        f"python main.py --mode prepro --data_file dataset/hotpot_train_v1_{number}.json --word_emb_file prepro_result/train/{number}/word_emb_file_{number}.json --char_emb_file prepro_result/train/{number}/char_emb_file_{number}.json --word2idx_file prepro_result/train/{number}/word2idx_file_{number}.json --char2idx_file prepro_result/train/{number}/char2idx_file_{number}.json --idx2word_file prepro_result/train/{number}/idx2word_file_{number}.json --idx2char_file prepro_result/train/{number}/idx2char_file_{number}.json --train_eval_file prepro_result/train/{number}/char2idx_file_{number}.json --para_limit 5000 --train_eval_file prepro_result/train/{number}/train_eval_{number}.json --data_split train"
    )

"""
python main.py --mode prepro --data_file dataset/hotpot_train_v1_0.json --word_emb_file prepro_result/train/0/word_emb_file_0.json --char_emb_file prepro_result/train/0/char_emb_file_0.json --word2idx_file prepro_result/train/0/word2idx_file_0.json --char2idx_file prepro_result/train/0/char2idx_file_0.json --idx2word_file prepro_result/train/0/idx2word_file_0.json --idx2char_file prepro_result/train/0/idx2char_file_0.json --train_eval_file prepro_result/train/0/char2idx_file_0.json --para_limit 5000 --train_eval_file prepro_result/train/0/train_eval_0.json --data_split train
python main.py --mode prepro --data_file dataset/hotpot_train_v1_1.json --word_emb_file prepro_result/train/1/word_emb_file_1.json --char_emb_file prepro_result/train/1/char_emb_file_1.json --word2idx_file prepro_result/train/1/word2idx_file_1.json --char2idx_file prepro_result/train/1/char2idx_file_1.json --idx2word_file prepro_result/train/1/idx2word_file_1.json --idx2char_file prepro_result/train/1/idx2char_file_1.json --train_eval_file prepro_result/train/1/char2idx_file_1.json --para_limit 5000 --train_eval_file prepro_result/train/1/train_eval_1.json --data_split train
python main.py --mode prepro --data_file dataset/hotpot_train_v1_2.json --word_emb_file prepro_result/train/2/word_emb_file_2.json --char_emb_file prepro_result/train/2/char_emb_file_2.json --word2idx_file prepro_result/train/2/word2idx_file_2.json --char2idx_file prepro_result/train/2/char2idx_file_2.json --idx2word_file prepro_result/train/2/idx2word_file_2.json --idx2char_file prepro_result/train/2/idx2char_file_2.json --train_eval_file prepro_result/train/2/char2idx_file_2.json --para_limit 5000 --train_eval_file prepro_result/train/2/train_eval_2.json --data_split train
"""


# print(train[1])
