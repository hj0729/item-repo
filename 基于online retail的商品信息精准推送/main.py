import Data_process
import Correlation_analysis
import ML
import pandas as pd


def main():
    main_data = Data_process.process_datas()

    while True:
        new_product_description = input("请输入推荐的商品名称 (输入 'exit' 退出): ").upper()

        if new_product_description == 'EXIT':
            break

        if new_product_description in main_data['Description'].values:
            Correlation_analysis.correlation_recommend(new_product_description)
        else:
            ML.ml_recommend(new_product_description)


if __name__ == '__main__':
    main()
