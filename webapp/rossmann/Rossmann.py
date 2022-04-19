import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann( object ):
    def __init__( self ):
        self.home_path=''
        self.competition_distance_scaler   = pickle.load( open( self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb') )
        self.weeks_in_promo2_scaler        = pickle.load( open( self.home_path + 'parameter/weeks_in_promo2_scaler.pkl', 'rb') )
        self.year_scaler                   = pickle.load( open( self.home_path + 'parameter/year_scaler.pkl', 'rb') )
        self.store_type_scaler             = pickle.load( open( self.home_path + 'parameter/store_type_scaler.pkl', 'rb') )
        
        
    def data_cleaning( self, df_ross ): 
        
        # renaming columns
        cols_snake = list( map( lambda x: inflection.underscore( x ), df_ross.columns ) )
        df_ross.columns = cols_snake


        ## Data Types
        df_ross['date'] = pd.to_datetime( df_ross['date'] ) # make sure it's datetime.

        # Fillout NA
        max_distance_in_df = df_ross['competition_distance'].max()
        df_ross['competition_distance'].fillna(value=max_distance_in_df * 5, inplace=True)

        # competition_open_since_month and competition_open_since_year
        """ For all NA values in this feature we have a competitor_distance associated with it. So, we are dealing with a 
        missing data, since there is a competitor nearby. To fill these NA values we will consider the data of the first 
        sell on each store."""

        df_ross['competition_open_since_year'] = df_ross.apply(first_sell_year, axis=1)
        df_ross['competition_open_since_month'] = df_ross.apply(first_sell_month, axis=1)

        # # promo2_since_year and promo2_since_week 
        """ Following the same strategy we took in competition_distance, we will set a far away date to the promo2_since, 
        in order to simulate that a store did not participate in promo2."""
        df_ross['promo2_since_year'].fillna(2050, inplace=True ) 
        df_ross['promo2_since_week'].fillna(1, inplace=True )

        # promo_interval 
        df_ross['promo_interval'].fillna(0, inplace=True )

        ## Change Data Types
        # competiton
        # changing from float to int
        df_ross['competition_open_since_month'] = df_ross['competition_open_since_month'].astype( int )
        df_ross['competition_open_since_year'] = df_ross['competition_open_since_year'].astype( int )

        df_ross['promo2_since_week'] = df_ross['promo2_since_week'].astype( int ) 
        df_ross['promo2_since_year'] = df_ross['promo2_since_year'].astype( int )
        
        return df_ross 


    def feature_engineering( self, df_ross ):

        # changing days_of_week
        week_map = {1: 'Monday',  2: 'Tuesday',  3: 'Wednesday',  4: 'Thursday',  5: 'Friday',  6: 'Saturday',  7: 'Sunday'}
        df_ross['day_of_week_name'] = df_ross['day_of_week'].map( week_map )

        # year
        df_ross['year'] = df_ross['date'].dt.year

        # month
        df_ross['month'] = df_ross['date'].dt.month

        # year week
        df_ross['year_week'] = df_ross['date'].dt.strftime( '%Y-%W' )

        # year month
        df_ross['year_month'] = df_ross['date'].dt.strftime('%Y-%m')


        # error value 1900 in competition_open_since_year
        """the 1900 value can be an error in previous parsing. while parsing, the 1900-01-01T00:00:00.000 is a default value"""
        df_ross['competition_open_since_year'] = df_ross.apply(year_correction, axis=1)

        # assortment
        df_ross['assortment'] = df_ross['assortment'].apply( lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended' )

        # state holiday
        df_ross['state_holiday'] = df_ross['state_holiday'].apply( lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day' )

        # promo since
        df_ross['promo2_since'] = df_ross['promo2_since_year'].astype( str ) + '-' + df_ross['promo2_since_week'].astype( str )
        df_ross['promo2_since'] = df_ross['promo2_since'].apply( lambda x: datetime.strptime( x + '-1', '%Y-%W-%w' ) )
        df_ross['weeks_in_promo2'] = ( ( df_ross['date'] - df_ross['promo2_since'] ) /7 ).apply( lambda x: x.days ).astype( int )


        # 3.0. PASSO 03 - FILTRAGEM DE VARIÁVEIS
        ## 3.1. Filtragem das Linhas
        df_ross = df_ross.query('open == 1 and sales > 0')

        ## 3.2. Selecao das Colunas
        df_ross.drop( ['open', 'promo_interval'], axis=1, inplace=True )
        
        return df_ross


    def data_preparation( self, df_ross ):

        ## Rescaling 
        # competition distance -> robust scaler
        df_ross['competition_distance'] = self.competition_distance_scaler.transform( df_ross[['competition_distance']].values )

        # year MinMax Scaler
        df_ross['year'] = self.year_scaler.transform( df_ross[['year']].values )
        
        # weeks in promo2 -> MinMax scaler 
        df_ross['weeks_in_promo2'] = self.weeks_in_promo2_scaler.transform( df_ross[['weeks_in_promo2']].values )

        ### Encoding
        # state_holiday - One Hot Encoding
        df_ross = pd.get_dummies( df_ross, prefix=['state_holiday'], columns=['state_holiday'] )

        # store_type - Label Encoding
        le = LabelEncoder()
        df_ross['store_type'] = le.fit_transform( df_ross['store_type'] )
        pickle.dump( le, open( 'parameter/store_type_scaler.pkl', 'wb') )

        # assortment - Ordinal Encoding
        assortment_dict = {'basic': 1,  'extra': 2, 'extended': 3}
        df_ross['assortment'] = df_ross['assortment'].map( assortment_dict )

        
        ### Nature Transformation
        # day of week
        df_ross['day_of_week_sin'] = df_ross['day_of_week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
        df_ross['day_of_week_cos'] = df_ross['day_of_week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

        # month
        df_ross['month_sin'] = df_ross['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
        df_ross['month_cos'] = df_ross['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

       
        cols_selected_boruta = ['store', 'promo', 'store_type', 'assortment', 'competition_distance',
        'competition_open_since_month', 'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year',
        'weeks_in_promo2', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']
        
        return df_ross[ cols_selected ]
    
    
    def get_prediction( self, model, original_data, test_data ):
        # prediction
        pred = model.predict( test_data )
        
        # join pred into the original data
        original_data['prediction'] = np.expm1( pred )
        
        return original_data.to_json( orient='records', date_format='iso' )