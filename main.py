import csv
import json
import numpy as np
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
    @return data_final: list of list of items grouped by customer_id, triplicates removed
    @return cust_id: list of customer_id
    @return prod_id: list of product_id
    """
    group, cust_id, prod_id = regroup(data_raw)
    # Only allow duplicates to take into account cases of repeated purchases
    # Items repeated more than twice in the list will not contribute to computation
    data_final = remove_triplicates(group)
    return data_final, cust_id, prod_id[0].tolist()


def regroup(data_raw):
    """
    Regroups products by customer_id

    @param data_raw: list of dictionaries with transaction properties
    @return ret: nested list of product_id's purchased by individual customers
    @return user_id_list: list of customer_id
    @return product_id_list: list of product_id
    """

    # Create data frame
    cols = list(data_raw[0].keys())
    data_values = []
    for item in data_raw:
        data_values.append(list(item.values()))
    df = pd.DataFrame(data_values, columns=cols)

    # Retrieve list of all products
    product_id_list = df['product_id'].tolist()
    # Group transactions by customer_id
    grouped = df.groupby('customer_id')
    user_id_list = []
    ret = []
    # Elements in ret and user_id_list have a common index
    for group in grouped:
        user_id_list.append(group[0])
        # Group all products purchased by individual users
        ret.append(group[1].get('product_id').tolist())
    return ret, user_id_list, product_id_list


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
    @return ret: list of eligible sets
    @return supp_data: support values for eligible sets
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
    # Calculate support values and add to list if over support threshold
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


def apriori(data_set, min_supp=0.1):
    """
    Generates set of frequent item sets that meet minimum support from data set

    @param data_set: data set
    @param min_supp: minimum support value
    @return r: generated list of frequent item sets
    @return supp_data: support values for frequent item sets
    """
    d_processed, _, _ = process(data_set)
    c1 = create_c1(d_processed)
    l1, supp_data = scan(d_processed, c1, min_supp)
    r = [l1]
    k = 2

    while len(r[k - 2]) > 0:
        c_k = apriori_gen(r[k - 2], k)
        l_k, supp_k = scan(d_processed, c_k, min_supp)
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


def calc_conf(freq_set, h, supp_data, brl, min_conf=0.1):
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
            print(freq_set-conseq, '-->', conseq, 'conf:', conf)
            brl.append((freq_set-conseq, conseq, conf))
            pruned_h.append(conseq)
    return pruned_h


def rules_from_conseq(freq_set, h, supp_data, brl, min_conf=0.1):
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


def calculate_probabilities(rules_list, purchases, customers_list, products_list):
    """
    Calculates purchase probabilities for customers based on set of rules

    @param rules_list: list of recommendation rules
    @param purchases: list of user purchases organized by customer_id
    @param customers_list: list of all customers
    @param products_list: list of all products
    @return matrix of probabilities
    """
    res = [[0 for x in range(len(products_list))] for y in range(len(customers_list))]
    # Iterate through all users and rules
    for user in range(len(purchases)):
        for rule in rules_list:
            # If all items from rule were purchased by user, add confidence values
            # of recommended products to probabilities matrix
            if rule[0].issubset(purchases[user]):
                recommended_products = rule[1]
                conf = rule[2]
                for prod in recommended_products:
                    res[user][products_list.index(prod)] += conf
    # Normalize results such that all probabilities for single user add up to 1
    for user in range(len(customers_list)):
        sum_vals = sum(res[user])
        for prod in range((len(products_list))):
            if res[user][prod] > 0:
                res[user][prod] = res[user][prod] / sum_vals
    return res


def write_to_csv(list_to_write, file_name):
    """
    Write list to CSV file

    @param list_to_write: list to write to CSV file
    @param file_name: name of file
    """
    with open(file_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        if type(list_to_write) is list:
            for item in list_to_write:
                writer.writerow([str(item)])
        else:
            for item in list_to_write:
                writer.writerow(str(item))

# Declare thresholds
min_support = 0.01
min_confidence = 0.1

# Load and process data
data = load('training_mixpanel.txt')
data_processed, customers, products = process(data)

# Run main algorithm and calculate probabilities
L, support_data = apriori(data_processed, min_support)
rules = generate_rules(L, support_data, min_confidence)
results = calculate_probabilities(rules, data_processed, customers, products)

# Save data to CSV file
# results.csv: m by n matrix where m is indexed by customers.csv
# and n is indexed by products_list.csv
np.savetxt("results.csv", results, '%5.2g', delimiter=",")
write_to_csv(customers, 'customers_list.csv')
write_to_csv(products, 'products_list.csv')
