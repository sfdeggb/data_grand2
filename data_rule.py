import datetime

def time_condition(callTime):
    """
    当月总天数除以当前天数
    """
    time_condition = datetime.datetime.strptime(callTime, '%Y-%m-%d %H:%M:%S')
    return time_condition.days/time_condition.daysinmonth

def products_classier(user_feature, products_features):
    user_tags = []
    
    is_gprs_over = user_feature['is_gprs_over']
    tcw_gprs_fee1 = user_feature['tcw_gprs_fee1']
    tcw_gprs_fee2 = user_feature['tcw_gprs_fee2'] 
    tcw_gprs_fee3 = user_feature['tcw_gprs_fee3'] 
    callTime = user_feature['callTime']
    gprs_all = user_feature['gprs_all']
    gprs_residua = user_feature['gprs_residua']

    #当月+连续三个月流量套餐
    if is_gprs_over==1 and tcw_gprs_fee1>0 and tcw_gprs_fee2>0 and tcw_gprs_fee3>0:
        user_tags.append('当月+连续三个月流量套餐')
    #当月+连续两个月流量套餐
    if is_gprs_over==1 and tcw_gprs_fee1>0 and tcw_gprs_fee2>0:
        user_tags.append('当月+连续两个月流量套餐')
    #当月+上月流量套餐
    if is_gprs_over==1 and tcw_gprs_fee1>0:
        user_tags.append('当月+上月流量套餐')
    #当月流量套餐
    if is_gprs_over==1:
        user_tags.append('当月流量套餐')
    #当月流量预超套+连续3个月流量超套
    time_condition = time_condition(callTime)
    if gprs_all*time_condition -  (gprs_all+gprs_residua)>0 and tcw_gprs_fee1 >0 and tcw_gprs_fee2 >0 and tcw_gprs_fee2 >0:
        user_tags.append('当月流量预超套+连续3个月流量超套')
    #当月超预套
    if gprs_all*time_condition - (gprs_all+gprs_residua)>0 and tcw_gprs_fee1 >0:
        user_tags.append('当月超预套')
    # 连续三个月超套
    if tcw_gprs_fee1 >0 and tcw_gprs_fee2 >0 and tcw_gprs_fee3 >0:
        user_tags.append('连续三个月超套')
    # 连续两个月超套
    if tcw_gprs_fee1 >0 and tcw_gprs_fee2 >0:
        user_tags.append('连续两个月超套')
    # 上月超套
    if tcw_gprs_fee1 >0:
        user_tags.append('上月超套')
    return user_tags

# 示例使用
user_feature_example = {
    "is_gprs_over": True,
    "tcw_gprs_fee1": 10,
    "tcw_gprs_fee2": 0,
    "tcw_gprs_fee3": 0,
    "callTime": "2024-11-20 20:23:32",
    "gprs_all": 20,
    "gprs_residua": 2,
    "app_usage": {"some_app": 60},
    "is省外流量超套": False,
    "recommendation_conversion_rate": 0.6,
    "product_data": 25,
    "user_data": 20,
    "promotion": True,
    "weekly_sales_rank": 1,
    "monthly_sales_rank": 2,
    "is_new": True,
    "price_per_unit": 0.5,
    "original_price": 1.0
}

products_features_example = []  # 产品特征

user_tags = products_classier(user_feature_example, products_features_example)
print(user_tags)