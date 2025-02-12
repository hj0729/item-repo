import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def read_file(file_path):
    # 读取JSON文件
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 初始化一个空列表来存储每个商品的记录
    product_records = []

    # 遍历外层字典
    for product_name, details in data.items():
        # 创建一个记录字典，包含商品名称、描述和顾客ID列表
        record = {
            'Product_Name': product_name,
            'Description': details['Description'],
            'Customer_IDs': details.get('Customer_ID', [])  # 确保是列表类型
        }
        # 将记录添加到列表中
        product_records.append(record)

    # 将列表转换为DataFrame
    product_df = pd.DataFrame(product_records)
    return product_df


def extract_features(dataframe, column_name):
    # 使用TF-IDF提取特征
    tfidf_vectorizer = TfidfVectorizer()
    feature_matrix = tfidf_vectorizer.fit_transform(dataframe[column_name])
    return feature_matrix, tfidf_vectorizer


def find_similar_products(new_description, feature_matrix, data_frame, vectorizer, top_n=5):
    # 将新商品描述转换为特征向量
    new_product_features = vectorizer.transform([new_description])
    # 计算余弦相似度
    similarity_scores = cosine_similarity(new_product_features, feature_matrix)
    # 获取最相似的商品索引
    similar_indices = similarity_scores.argsort()[0, ::-1][:top_n]
    # 获取最相似商品的顾客ID列表，并合并去重
    similar_customer_ids = set()
    for index in similar_indices:
        similar_customer_ids.update(data_frame.iloc[index]['Customer_IDs'])
    return list(similar_customer_ids)


def ml_recommend(main_description):
    # 读取数据
    df = read_file('recommended_customers.json')

    # Description列包含商品描述
    X, vec = extract_features(df, 'Description')

    # 输入其它的商品名称
    new_product_description = main_description

    # 找到可能对该新商品感兴趣的顾客
    similar_customers = find_similar_products(new_product_description, X, df, vec)

    print(f"可能对新商品 '{new_product_description}' 感兴趣的顾客ID: {similar_customers}")
