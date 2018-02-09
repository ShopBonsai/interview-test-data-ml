import json
import pandas as pd


def load(file_name):
    """
    Loads product properties from JSON file
    @param file_name: name of JSON file
    @return: list of dictionaries containing transaction properties
    """
    with open(file_name) as file:
        data_raw = json.load(file)
    ret = []
    for event in data_raw:
        ret.append(event['properties'])
    return ret


def process(data_raw):
    """
    Prepares data for analysis

    @param data_raw: list of dictionaries with transaction properties
    @return: list of list of items grouped by customer_id, triplicates removed
    """
    group = regroup(data_raw)
    ret = remove_triplicates(group)
    return ret


def regroup(data_raw):
    """
    Regroups products by customer_id

    @param data_raw: list of dictionaries with transaction properties
    @return: list of lists of items grouped by customer_id
    """
    cols = list(data_raw[0].keys())
    data_values = []
    for item in data_raw:
        data_values.append(list(item.values()))
    df = pd.DataFrame(data_values, columns=cols)
    grouped = df.groupby('customer_id')
    ret = []
    for group in grouped:
        ret.append(group[1].get('product_id').tolist())
    return ret


def remove_triplicates(data_raw):
    """
    Removes triplicates

    @param data_raw: list of lists of items grouped by customer_id
    @return: original list of lists with triplicates removed
    """
    ret = []
    for item in data_raw:
        seen = []
        dupl = []
        for product in item:
            if product not in seen:
                seen.append(product)
            elif product not in dupl:
                dupl.append(product)
        ret.append(seen + dupl)
    return ret


def create_c1(data_raw):
    """
    Creates initial set of individual items

    @param data_raw: list of lists of items grouped by customer_id
    @return: list of frozensets of 1 item
    """
    c_1 = []
    for customer in data_raw:
        for product in customer:
            if not [product] in c_1:
                c_1.append([product])

    c_1.sort()
    return list(map(frozenset, c_1))


def scan(data_raw, ck, min_supp):
    """
    Scans candidate sets in data set for items that meet the minimum support

    @param data_raw: data set
    @param ck: list of candidate sets
    @param min_supp: minimum support
    @return: list of eligible sets and their support values
    """
    ss_cnt = {}
    for tid in data_raw:
        for can in ck:
            if can.issubset(tid):
                if can not in ss_cnt:
                    ss_cnt[can] = 1
                else:
                    ss_cnt[can] += 1
    num_items = float(len(data_raw))
    ret = []
    supp_data = {}
    for key in ss_cnt:
        support = ss_cnt[key]/num_items
        if support >= min_supp:
            ret.insert(0, key)
        supp_data[key] = support
    return ret, supp_data


def apriori_gen(lk, size):
    """
    Generates candidate sets from frequent item sets

    @param lk: list of frequent item sets
    @param size: size of item sets
    @return: generated candidate sets
    """
    ret = []
    len_lk = len(lk)
    for i in range(len_lk):
        for j in range(i+1, len_lk):
            l_1 = list(lk[i])[:size-2]
            l2 = list(lk[j])[:size-2]
            l_1.sort()
            l2.sort()
            if l_1 == l2:
                ret.append(lk[i] | lk[j])
    return ret


def apriori(data_set, min_supp=0.01, min_l1_supp=0.1):
    """
    Generates set of frequent item sets that meet minimum support from data set

    @param data_set: data set
    @param min_supp: minimum support value
    @param min_l1_supp: minimum support value for single items
    @return: generated list of frequent item sets with support values
    """
    data_processed = process(data_set)
    c1 = create_c1(data_processed)
    l1, supp_data = scan(data_processed, c1, min_l1_supp)
    r = [l1]
    k = 2

    while len(r[k - 2]) > 0:
        c_k = apriori_gen(r[k - 2], k)
        l_k, supp_k = scan(data_processed, c_k, min_supp)
        supp_data.update(supp_k)
        r.append(l_k)
        k += 1
    return r, supp_data


def generate_rules(item_set, supp_data, min_conf):
    """
    Generates set of rules that meet minimum confidence value from list
    of frequent item set

    @param item_set: list of frequent item sets
    @param supp_data: dictionary with support data for item_set
    @param min_conf: minimum confidence value
    @return: generated rules
    """
    rules_list = []
    for i in range(1, len(item_set)):  # only get the sets with two or more items
        for freq_set in item_set[i]:
            h1 = [frozenset([item]) for item in freq_set]
            if i > 1:
                rules_from_conseq(freq_set, h1, supp_data, rules_list, min_conf)
            else:
                calc_conf(freq_set, h1, supp_data, rules_list, min_conf)
    return rules_list


def calc_conf(freq_set, h, supp_data, brl, min_conf=0.7):
    """
    Calculates confidence values for set of frequent items

    @param freq_set: list of frequent item sets
    @param h: list of items that could be on RHS of rule
    @param supp_data: dictionary with support data
    @param brl: global rules list
    @param min_conf: minimum confidence value
    @return: generated rules
    """
    pruned_h = []
    for conseq in h:
        conf = supp_data[freq_set]/supp_data[freq_set-conseq]
        if conf >= min_conf:
            print (freq_set-conseq,'-->',conseq,'conf:',conf)
            brl.append((freq_set-conseq, conseq, conf))
            pruned_h.append(conseq)
    return pruned_h


def rules_from_conseq(freq_set, h, supp_data, brl, min_conf=0.7):
    """
    Generates additional rules from initial data set

    @param freq_set: list of frequent item sets
    @param h: list of items that could be on RHS of rule
    @param supp_data: dictionary with support data
    @param brl: global rules list
    @param min_conf: minimum confidence value
    """
    m = len(h[0])
    if len(freq_set) > (m + 1):
        hmp1 = apriori_gen(h, m + 1)
        hmp1 = calc_conf(freq_set, hmp1, supp_data, brl, min_conf)
        if len(hmp1) > 1:
            rules_from_conseq(freq_set, hmp1, supp_data, brl, min_conf)


min_support = 0.001
min_l1_support = 0.05
min_confidence = 0.1
data = load('training_mixpanel.txt')
L, support_data = apriori(data, min_support, min_l1_support)
rules = generate_rules(L, support_data, min_confidence)
