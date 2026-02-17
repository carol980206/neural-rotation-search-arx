def gen_diffs(diff_indexes, word_size=32, word_num=4):
    diffs = []
    for diff_index in diff_indexes:
        diff = [0 for _ in range(word_num)]
        for idx in diff_index:
            diff[word_num - 1 - (idx // word_size)] |= 1 << (idx % word_size)
        diff = tuple(diff)
        diffs.append(diff)
    return diffs

def gen_diff(diff_index, word_size=32, word_num=4):
    diff = [0 for _ in range(word_num)]
    for idx in diff_index:
        diff[word_num - 1 - (idx // word_size)] |= 1 << (idx % word_size)
    return tuple(diff)

def gen_diff_index(diff, word_size=32, word_num=4):
    diff_index = []
    for i in range(word_num * word_size):
        if ((diff[word_num - 1 - (i // word_size)] >> (i % word_size)) & 1) == 1:
            diff_index.append(i)
    return diff_index