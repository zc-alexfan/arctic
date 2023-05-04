import math


def chunks_by_len(L, n):
    """
    Split a list into n chunks
    """
    num_chunks = int(math.ceil(float(len(L)) / n))
    splits = [L[x : x + num_chunks] for x in range(0, len(L), num_chunks)]
    return splits


def chunks_by_size(L, n):
    """Yield successive n-sized chunks from lst."""
    seqs = []
    for i in range(0, len(L), n):
        seqs.append(L[i : i + n])
    return seqs


def unsort(L, sort_idx):
    assert isinstance(sort_idx, list)
    assert isinstance(L, list)
    LL = zip(sort_idx, L)
    LL = sorted(LL, key=lambda x: x[0])
    _, L = zip(*LL)
    return list(L)


def add_prefix_postfix(mydict, prefix="", postfix=""):
    assert isinstance(mydict, dict)
    return dict((prefix + key + postfix, value) for (key, value) in mydict.items())


def ld2dl(LD):
    assert isinstance(LD, list)
    assert isinstance(LD[0], dict)
    """
    A list of dict (same keys) to a dict of lists
    """
    dict_list = {k: [dic[k] for dic in LD] for k in LD[0]}
    return dict_list


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    seqs = []
    for i in range(0, len(lst), n):
        seqs.append(lst[i : i + n])
    seqs_chunked = sum(seqs, [])
    assert set(seqs_chunked) == set(lst)
    return seqs
