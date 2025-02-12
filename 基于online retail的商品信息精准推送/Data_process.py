import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def data_visualization(data):  # 数据可视化
    large = 22
    med = 16
    params = {
        'axes.titlesize': large,
        'legend.fontsize': med,
        'figure.figsize': (16, 10),  # 默认图表大小
        'axes.labelsize': med,
        'xtick.labelsize': med,
        'ytick.labelsize': med,
        'figure.titlesize': large
    }
    plt.rcParams.update(params)
    sns.set_style('whitegrid')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 负号"-"正确显示

    # 获取每个国家的产品数量
    country_product_count = data['Country'].value_counts()

    plt.figure(figsize=(14, 8))  # 指定图表大小
    country_product_count.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Product Quantity by Country')
    plt.xlabel('Country')
    plt.ylabel('Number of Products')
    plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，并右对齐
    plt.tight_layout()  # 调整布局以防止标签被裁剪
    plt.show()


def read_text(filename):  # 读取数据并显示数据前几行预览
    pd.set_option('display.max_columns', None)
    df_online = pd.read_excel(filename)
    # print(df_online.head())
    # print(df_online.info())
    return df_online


def process_miss_value(data):
    # print("缺失值数量：")
    # print(data.isnull().sum())
    # print("重复值数量")
    # print(data.duplicated().sum())

    # 复制数据以避免警告
    data = data.copy()

    # 删除发票编号以'C'开头的订单
    data['Invoice'] = data['Invoice'].astype(str)  # 将 'Invoice' 列转换为字符串类型
    data = data[~data['Invoice'].str.startswith('C')]

    # 删除多余重复数据，保留一条
    data.drop_duplicates(inplace=True)

    # 明确使用 .loc 来避免警告
    data.loc[:, 'Quantity'] = data['Quantity'].fillna(data['Quantity'].mean())
    data.loc[:, 'Price'] = data['Price'].fillna(data['Price'].mean())
    data['Customer ID'] = data['Customer ID'].fillna('Unknown')

    cols_to_fill = ['StockCode', 'Description', 'InvoiceDate', 'Country']
    data.loc[:, cols_to_fill] = data[cols_to_fill].fillna('Null')

    return data


def commodity_statistic(data):  # 统计商品信息
    # 统计不同商品种类数量和名称
    model = data['StockCode'].nunique()
    model_names = data['Description'].unique()
    print("共有" + str(model) + "种商品\n")

    # 最畅销的15种商品
    best_seller = data.groupby('StockCode').size().reset_index(name='count')

    # 获取 Description，将 best_seller 与原始数据结合
    description_map = data[['StockCode', 'Description']].drop_duplicates().set_index('StockCode')
    best_seller_with_desc = best_seller.join(description_map, on='StockCode', how='left')

    # 获取前15个
    top_15 = best_seller_with_desc.head(15)

    # 绘制最畅销的15种商品的条形图
    plt.figure(figsize=(14, 8))  # 指定图表大小
    sns.barplot(x='StockCode', y='count', data=top_15)
    plt.title('最畅销的15种商品')
    plt.grid(True)
    plt.xlabel('商品代码')
    plt.ylabel('销量')
    plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，并右对齐
    plt.tight_layout()  # 调整布局以防止标签被裁剪

    # 获取前15个商品的 Description
    top_15_names = top_15['Description'].tolist()
    print('由销量排名，排名前15的畅销单品为：')
    for i in range(0, 15, 5):
        print(top_15_names[i:i + 5])

    # 显示图表
    plt.show()


def process_datas():
    online_data = read_text('online_retail_II.xlsx')
    process_data = process_miss_value(online_data)
    data_visualization(process_data)
    commodity_statistic(process_data)
    return process_data
