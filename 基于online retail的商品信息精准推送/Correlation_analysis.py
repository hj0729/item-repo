import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
import Data_process
import json


def generate_basket(data):  # 生成购物篮
    baskets = data.groupby('Invoice')['Description'].apply(lambda x: list(x.astype(str).str.strip().str.upper()))
    te = TransactionEncoder()
    baskets_tf = te.fit_transform(baskets)
    df_basket = pd.DataFrame(baskets_tf, columns=te.columns_, dtype=bool)
    return df_basket


def correlation_analysis(df_baskets):  # 关联规则分析
    frequent_itemsets = apriori(df_baskets, min_support=0.02, use_colnames=True)
    rules = association_rules(frequent_itemsets, num_itemsets=len(frequent_itemsets), metric='confidence',
                              min_threshold=0.01)
    # print(frequent_itemsets.head())  # 调试

    rules = rules[rules.lift >= 1.0]
    rules.rename(columns={'antecedents': 'lhs', 'consequents': 'rhs', 'support': 'sup', 'confidence': 'conf'},
                 inplace=True)
    rules = rules[['lhs', 'rhs', 'sup', 'conf', 'lift']]

    return rules


def get_customer_ids_for_product(data, product):
    # 找到购买过该商品的顾客ID
    customer_ids = data[data['Description'].str.contains(product, case=False, na=False)]['Customer ID'].unique()
    return customer_ids


def product_recommendation(rules, process_data, main_product):
    product_set = frozenset({main_product})
    lhs_matches = rules[rules['lhs'].apply(lambda x: product_set.issubset(x))]
    rhs_matches = rules[rules['rhs'].apply(lambda x: x == product_set)]

    if lhs_matches.empty and rhs_matches.empty:
        customer_ids = get_customer_ids_for_product(process_data, main_product)
        if len(customer_ids) > 0:
            print(f"虽然没有直接的关联规则，但推荐购买 {main_product} 的顾客ID：{customer_ids}")
            record_data(main_product, customer_ids, 'recommended_customers.json')
        else:
            print("抱歉，输入的商品不存在，且没有顾客购买记录，请重新输入。")
    elif not rhs_matches.empty:
        # 处理 rhs_matches 不为空的情况
        customer_ids = get_customer_ids_for_product(process_data, rhs_matches)  # 需要定义这个函数
        print(f"基于 RHS 规则推荐购买 {main_product} 的顾客ID：{customer_ids}")
        record_data(main_product, customer_ids, 'recommended_customers.json')
    else:
        # 处理 lhs_matches 不为空而 rhs_matches 为空的情况
        print("没有直接的 RHS 规则，使用 LHS 规则进行推荐：")
        # 这里添加处理 lhs_matches 的逻辑


def record_data(input_product, customer_ids, file_path='recommended_customers.json'):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    # 创建一个包含商品描述和顾客ID的字典
    record = {
        'Description': input_product,
        'Customer_ID': customer_ids.tolist()  # 将顾客ID转换为列表
    }

    # 使用商品描述作为键来存储数据
    data[input_product] = record

    # 使用 indent 参数来格式化 JSON 输出
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)  # 添加 indent=4 以增加可读性

    print(f"数据已成功记录到文件 {file_path}")


def correlation_recommend(main_product):
    process_data = Data_process.process_miss_value(Data_process.read_text('online_retail_II.xlsx'))
    df = generate_basket(process_data)
    product_rules = correlation_analysis(df)
    product_recommendation(product_rules, process_data, main_product)
