import os
import json
import requests 
import tempfile
import telegram
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from flask import Flask, request, Response

# constants
TOKEN = os.environ['rossman_token_bot']

text_start = '''Hello There! \U0001F604  

                I am robot that will predict the sales of any Rossmann store. If you give me a store number (/num), I will give you the prediction. If you are not sure whether a store exists or not, I can sample 10 stores for you, if you type /list. 
                
                If you want to check the overall result of the machine learning model, type /model.
                
                If you want to know more about my creator, just type /author. '''


text_author = """ Hi \U0001F44B
                  This is a bot created by Rodrigo Maranhão to predict sales of Rossmann stores, solving an forecast problem using Machine Learning and a dataset from Kaggle.
                  If you want to understand the context and know more about the project, please visit my GitHub page: https://github.com/rodrigomm92/rossmann_forecast_sales
                  If you are passionate about the universe of Data Science, contact with me through my Linkedin: https://www.linkedin.com/in/rodrigomaranhaomonteiro/ 
                  Or e-mail me at: rodrigomaranhao.m@gmail.com."""

text_model = """ The choosed model was a XGBoost Regressor. This model has an overall mean absolute percentage error (MAPE) of 9.9%. The chart below shows the comparison of the real sales and predicted sales:"""

bot = telegram.Bot(TOKEN)

        
def send_message( chat_id, text):
    url = 'https://api.telegram.org/bot{}/'.format( TOKEN ) 
    url = url + 'sendMessage?chat_id={}'.format( chat_id )
        
    r = requests.post( url, json={'text':text} ) 
    print('Status Code {}'.format(r.status_code) )
    
    return None

def send_chat_action( chat_id, action ):
    url = 'https://api.telegram.org/bot{}/'.format( TOKEN ) 
    url = url + 'sendChatAction'
        
    r = requests.post( url, json={'chat_id':chat_id, 'action':action} ) 
    print('Status Code {}'.format(r.status_code) )
    
    return None
    
def load_dataset( store_id ):
    # loading test dataset
    df10 = pd.read_csv( 'test.csv' )
    df_store_raw = pd.read_csv( 'store.csv' )

    # merge test dataset + store
    df_test = pd.merge( df10, df_store_raw, how='left', on='Store' )

    # choose store for prediction
    df_test = df_test[df_test['Store'] == store_id]
    
    if not df_test.empty:
        # remove closed days
        df_test = df_test[df_test['Open'] != 0]
        df_test = df_test[~df_test['Open'].isnull()]
        df_test = df_test.drop( 'Id', axis=1 )
        # convert Dataframe to json
        data = json.dumps( df_test.to_dict( orient='records' ) )
    else:
        data = 'error'
    
    return data

def predict( data ):

    # API Call
    telegram.constants.CHATACTION_TYPING
    url = 'https://my-rossmann-model-test.herokuapp.com/rossmann/predict'
    header = {'Content-type': 'application/json' } 
    data = data

    r = requests.post( url, data=data, headers=header )
    print( 'Status Code {}'.format( r.status_code ) )

    d1 = pd.DataFrame( r.json(), columns=r.json()[0].keys() )
    
    return d1

def store_list(chat_id):
    df_store_raw = pd.read_csv( 'store.csv' )
    stores = df_store_raw.sample(10)['Store'].unique()
    message = 'Example of stores: ' + str(stores)
    send_message(chat_id, message )
    return None


def std_font(ax1, title, xlabel, ylabel):
    ax1.set_title(title, loc='left', fontdict={'fontsize': 18}, pad=20)
    ax1.set_xlabel(xlabel, fontdict={'fontsize': 12, 'style': 'italic'})
    ax1.set_ylabel(ylabel, fontdict={'fontsize': 12, 'style': 'italic'})
    return None

def send_plots(chat_id, df):
    df.loc[:,'date'] = pd.to_datetime( df['date'] )
    df.loc[:,'year_month_day'] = df['date'].dt.strftime( '%Y-%m-%d' )
    fig= plt.figure(figsize=(10,6), tight_layout={'pad':2.0})
    plt.subplot( 2, 1, 2 )
    ax1 = sns.lineplot(data=df.sort_values('year_month_day'), x='year_month_day', y='prediction')
    std_font(ax1, 'Predicted Sales by Day',
         '', 'Income (US$)')
    a = plt.xticks(rotation=60)

    plt.subplot( 2, 1, 1 )
    week_map = {1: 'Monday',  2: 'Tuesday',  3: 'Wednesday',  4: 'Thursday',  5: 'Friday',  6: 'Saturday',  7: 'Sunday'}
    df['day_of_week'] = df['day_of_week'].map( week_map )
    df_predict = df[['day_of_week','prediction']].groupby('day_of_week').sum().sort_values('prediction').reset_index()
    ax2 = sns.barplot(data=df_predict, x='day_of_week', y='prediction') 
    std_font(ax2, 'Sum of Predicted Sales by Day of Week', '', 'Income (US$)')
            
    temp_dir = tempfile.TemporaryDirectory()
    path1 = temp_dir.name + '/' + 'imagem.png'           
    fig.savefig(path1)
    bot.send_photo(chat_id, photo=open(path1,'rb'))
    temp_dir.cleanup()
    return None

    
def parse_message( message ):
    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']
    
    if store_id[0] != '/':
        return chat_id, 'error'
    
    store_id = store_id.replace('/','')
    
    try:
        store_id = int(store_id)
        
    except ValueError:
        print('Message Error!')
    return chat_id, store_id
    
# initialize API
app = Flask( __name__ )

        

@app.route( '/', methods=['GET','POST'] )
def index():
    if request.method == 'POST':
        message = request.get_json()
        chat_id, store_id = parse_message( message )
        send_chat_action( chat_id, 'typing' )
        if store_id == 'start':
            send_message(chat_id, text_start )
            return Response('OK',status=200)
        
        if store_id == 'author':
            send_message(chat_id, text_author )
            return Response('OK',status=200)
        
        if store_id == 'num':
            message = 'Use this command to select a store. Example: /12'
            send_message(chat_id, message )
            return Response('OK',status=200)
        
        if store_id == 'list':
            store_list(chat_id)
            return Response('OK',status=200)
        
        if store_id == 'model':
            send_message(chat_id, text_model )
            bot.send_photo(chat_id, photo=open('final_model.png','rb'))
            return Response('OK',status=200)
        
        if store_id != 'error':
            # loading data
            data = load_dataset(store_id)           
            
            if data != 'error':
                # prediction
                d1 = predict(data)
                
                # calculation
                d2 = d1[['store', 'prediction']].groupby( 'store' ).sum().reset_index()

                # send message
                msg = 'Store Number {} will sell R${:,.2f} in the next 6 weeks:'.format( 
                            d2['store'].values[0], 
                            d2['prediction'].values[0] ) 
                
                send_message(chat_id, msg)
                send_plots(chat_id, d1)
                return Response('OK',status=200)
                
            else:
                send_message(chat_id, 'Store Not Available. Please select other store, or type /list to get examples of existing stores')
                return Response('OK',status=200)
              
        else:
            send_message(chat_id,'Not recognized command.')
            return Response('OK',status=200)
        
    else:
        return 'Rossmann Telegram BOT'

if __name__ == '__main__':
    port = os.environ.get('PORT',5000)
    app.run( host='0.0.0.0',port=port ) 