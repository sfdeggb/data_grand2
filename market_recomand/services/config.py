model_path = './model/gbdt_model.pkl'
label_encoder_path = './model/label_encoder.pkl'
standard_scaler_path = './model/standard_scaler.pkl'

sparse_features = ['class_name','goods_class_name','goods_class_id','skuid','mobile','prov_code']
dense_features = ['c_price', 'line_price', 'goods_7day_views', 'goods_14day_views', 'goods_30day_views', 
                  'goods_7day_sales', 'goods_14day_sales', 'goods_30day_sales', 
                  'u2i_7days_view_count', 'u2i_type1_7days_view_count', 
                  'u2i_type2_7days_view_count', 'u2i_14days_view_count', 'u2i_type1_14days_view_count', 
                  'u2i_type2_14days_view_count', 'u2i_30days_view_count', 'u2i_type1_30days_view_count', 
                  'u2i_type2_30days_view_count', 'u2i_60days_view_count', 'u2i_type1_60days_view_count', 
                  'u2i_type2_60days_view_count', 'u2i_1days_click_count', 'u2i_type1_1days_click_count', 
                  'u2i_type2_1days_click_count', 'u2i_7days_click_count', 'u2i_type1_7days_click_count', 
                  'u2i_type2_7days_click_count', 'u2i_14days_click_count', 'u2i_type1_14days_click_count', 
                  'u2i_type2_14days_click_count', 'u2i_30days_click_count', 'u2i_type1_30days_click_count', 
                  'u2i_type2_30days_click_count', 'u2i_60days_click_count', 'u2i_type1_60days_click_count', 
                  'u2i_type2_60days_click_count', 'u2i_7days_purchase_count', 'u2i_type1_7days_purchase_count', 
                  'u2i_type2_7days_purchase_count', 'u2i_14days_purchase_count', 'u2i_type1_14days_purchase_count', 
                  'u2i_type2_14days_purchase_count', 'u2i_30days_purchase_count', 'u2i_type1_30days_purchase_count', 
                  'u2i_type2_30days_purchase_count', 'u2i_60days_purchase_count', 'u2i_type1_60days_purchase_count', 
                  'u2i_type2_60days_purchase_count']
