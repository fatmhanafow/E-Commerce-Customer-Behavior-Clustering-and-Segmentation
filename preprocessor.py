import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.city_encoder = LabelEncoder()
        self.features_to_scale = None
        self.current_date = None
        
    def fit(self, df):
        # Learns the necessary parameters to transform the data.
        # data in order level -> customer level
        customer_df = self._create_customer_features(df)
        
        # Learning encoding for the city
        self.city_encoder.fit(customer_df['city_name_fa'])
        
        # Determine the columns that need to be scaled
        self.features_to_scale = customer_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # remove the customer ID from the list.
        if 'ID_Customer' in self.features_to_scale:
            self.features_to_scale.remove('ID_Customer')

        # Learning for Standardization
        self.scaler.fit(customer_df[self.features_to_scale])

        return self

    def transform(self, df):
        # function applies the learned transformations to the data.
        # Creating Customer Level Features
        customer_df = self._create_customer_features(df)
        
        # Convert City to Number
        customer_df['city_name_fa'] = self.city_encoder.transform(customer_df['city_name_fa'])
        
        # Standardization of numerical features
        scaled_features = self.scaler.transform(customer_df[self.features_to_scale])
        
        # Convert the result to DataFrame
        customer_df[self.features_to_scale] = scaled_features

        return customer_df

    def _create_customer_features(self, df):
        df_temp = df.copy()
        df_temp['DateTime_CartFinalize'] = pd.to_datetime(df_temp['DateTime_CartFinalize'])
        
        # We consider the latest date as "today".
        self.current_date = df_temp['DateTime_CartFinalize'].max()

        # Group by customer and create new features
        customer_features = df_temp.groupby('ID_Customer').agg(
            order_count=('ID_Order', 'nunique'),
            unique_item_count=('ID_Item', 'nunique'),
            total_quantity=('Quantity_item', 'sum'),
            total_spent=('Amount_Gross_Order', 'sum'),
            max_order_value=('Amount_Gross_Order', 'max'),
            last_order_date=('DateTime_CartFinalize', 'max'),
            city_name_fa=('city_name_fa', 'first')
        ).reset_index()

        # Create Derived Features
        # RFM (Recency, Frequency, Monetary)
        
        customer_features['average_order_value'] = customer_features['total_spent'] / customer_features['order_count']
        customer_features['average_quantity_per_order'] = customer_features['total_quantity'] / customer_features['order_count']
        customer_features['item_variety_ratio'] = customer_features['unique_item_count'] / customer_features['total_quantity']
        
        # Replacing infinite values with 0 (without using inplace)
        customer_features['item_variety_ratio'] = customer_features['item_variety_ratio'].replace([np.inf, -np.inf], 0)

        # create time features (Recency)
        customer_features['days_since_last_order'] = (self.current_date - customer_features['last_order_date']).dt.days
        customer_features = customer_features.drop(columns=['last_order_date'])

        return customer_features
