import numpy as np

sample = " if you want you"
char_set = list(set(sample))
char_dic = {c: i for i, c in enumerate(char_set)}

dic_size = len(char_dic)
hidden_size = len(char_dic)
learning_rate = 0.1

sample_idx = [char_dic[c] for c in sample]
x_data = [sample_idx[:-1]]
x_one_hot = [np.eye(dic_size)[x] for x in x_data]
print(x_one_hot)