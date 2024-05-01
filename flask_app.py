import firebase_admin
from firebase_admin import credentials, firestore
import tensorflow as tf
from keras.models import load_model
import os
import cv2 
import numpy as np
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, ImageMessage ,TextSendMessage, TextMessage, FollowEvent, FlexSendMessage, PostbackEvent, ImageSendMessage
from linebot.exceptions import InvalidSignatureError, LineBotApiError
import google.generativeai as genai
from datetime import datetime
import requests
import json
import os
import dotenv
dotenv.load_dotenv()
app = Flask(__name__)

CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)
current_path = os.getcwd()

model = load_model(
    os.path.join(current_path, "model\EfficientNetB0_version1.h5"),
    compile=False,
)

#Uri สำหรับรูป Menu and Instruction Image (ใช้เป็น https เท่านั้น) 
MENU_GUIDE = 'https://drive.google.com/uc?id=15SRVDEXDn3VyaJOOnmOP-yCGFs3NZ1Tl' 
INSTRUCTION_RES = 'https://drive.google.com/uc?id=1yyp888OPs2__xrtRxFHCFS9f01HA7wnE'

api_key =os.getenv("APIKEY")
generation_config = {
        "temperature": 0.1,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 5000,
    }

safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
   ]

datadict = {
    0: 'Banana in coconut milk (Kluay Buat Chi)',
    1: 'Boat Noodles with pork blood (Kuay Teow Reua)',
    2: 'Cabbage with Fish Sauce (Galam Plee Pad Nam Pla)',
    3: 'Chicken Biryani (Khao Mok Gai)',
    4: 'Chicken Green Curry (Gaeng Kiew Wan)',
    5: 'Chicken Panang Curry (Panang Gai)',
    6: 'Coconut Rlce pancake (Kanom Krok)',
    7: 'Crispy pork with Chinese broccoli and oyster sauce (Kana Mhoo Krob)',
    8: 'Egg and pork in sweet brown sauce (Moo Palo)',
    9: 'Egg custard in pumpkin (Sankaya Faktong)',
    10: 'Fired Rice (Khao Pad)',
    11: 'Fried Boiled Egg with Tamarind Sauce (Son-in-law_s Eggs)',
    12: 'Fried Mussel Pancakes (Hoi Tod)',
    13: 'Fried fish-paste balls (Tod Mun)',
    14: 'Fried noodles in soy sauce (Pad see ew)',
    15: 'Garlic and Pepper Pork with Rice (Khao Mhoo Tod Gratiem)',
    16: 'Glass noodle salad (Yam Woon Sen)',
    17: 'Golden egg yolk threads (Foi Thong)',
    18: 'Grass jelly (Chao Kuay)',
    19: 'Grilled River Prawns (Goong Phao)',
    20: 'Grilled pork (Mhoo Ping)',
    21: 'Mango Sticky Rice (Khao Niew Ma Muang)',
    22: 'Minced Pork Omelette (Kai Jeow Moo Saap)',
    23: 'Noodle with chicken curry (Kao Soi)',
    24: 'Pad Thai',
    25: 'Papaya Salad (Somtam)',
    26: 'Pork Satay with Peanut Sauce',
    27: 'Rice Noodles with Fish Curry Sauce (Kanom Jeen Nam Ya)',
    28: 'Shrimp Paste Fried Rice (Khao Kluk Kaphi)',
    29: 'Sour Soup with Shrimp Mixed Veggies (Kaeng Som Kung)',
    30: 'Spicy chicken curry in coconut milk (Tom Kha Gai)',
    31: 'Spicy minced pork salad(Larb mhoo)',
    32: 'Steamed fish with curry paste (Hor mok phra)',
    33: 'Stewed pork legs with rice (Khao Kha Mhoo)',
    34: 'Stir Fried Long Eggplant with Pork (Makheua Yao Pad mhoo)',
    35: 'Stir Fried Pork with Holy Basil and Chilies (Pad ka praow)',
    36: 'Stir fried chicken with cashew nuts',
    37: 'Stir fried clams with roasted chili paste (Hoyrai pad phrik pao)',
    38: 'Stir fried rice noodles with chicken (Kuaitiao khua kai)',
    39: 'Stir-Fried Beef with Yellow Curry Paste (KuaKling)',
    40: 'Stir-Fried Pumpkin with Eggs (Faktong Pad Kai)',
    41: 'Stir-fried crab with curry powder (BooPadPongali)',
    42: 'Stir-fried morning glory with red fire (Pak Boong Fai Dang)',
    43: 'Stuffed bitter gourd broth (Gaeng Jued Mara)',
    44: 'Tapioca balls with pork filling (Sakhu sai mu)',
    45: 'Thai Baked Shrimp with Glass Noodle (Goong Ob Woon Sen)',
    46: 'Thai Massaman Curry Chicken (Massaman Gai)',
    47: 'Thai layer dessert (Khanom Chan)',
    48: 'Thai spicy and sour soup (Tom yum Goong)',
    49: 'Yentafo'
}

#Initialize firebase app
if not firebase_admin._apps:
    #Secretkey Service account
    cred = credentials.Certificate(os.path.join(current_path, 'aroi-linebot-firebase-adminsdk-cdrok-0eb2b1e996.json'))
    firebase_admin.initialize_app(cred)
else:
    firebase_admin.get_app()
db = firestore.client()

#data
datadict = {
    0: 'Banana in coconut milk (Kluay Buat Chi)',
    1: 'Boat Noodles with pork blood (Kuay Teow Reua)',
    2: 'Cabbage with Fish Sauce (Galam Plee Pad Nam Pla)',
    3: 'Chicken Biryani (Khao Mok Gai)',
    4: 'Chicken Green Curry (Gaeng Kiew Wan)',
    5: 'Chicken Panang Curry (Panang Gai)',
    6: 'Coconut Rlce pancake (Kanom Krok)',
    7: 'Crispy pork with Chinese broccoli and oyster sauce (Kana Mhoo Krob)',
    8: 'Egg and pork in sweet brown sauce (Moo Palo)',
    9: 'Egg custard in pumpkin (Sankaya Faktong)',
    10: 'Fired Rice (Khao Pad)',
    11: 'Fried Boiled Egg with Tamarind Sauce (Son-in-law_s Eggs)',
    12: 'Fried Mussel Pancakes (Hoi Tod)',
    13: 'Fried fish-paste balls (Tod Mun)',
    14: 'Fried noodles in soy sauce (Pad see ew)',
    15: 'Garlic and Pepper Pork with Rice (Khao Mhoo Tod Gratiem)',
    16: 'Glass noodle salad (Yam Woon Sen)',
    17: 'Golden egg yolk threads (Foi Thong)',
    18: 'Grass jelly (Chao Kuay)',
    19: 'Grilled River Prawns (Goong Phao)',
    20: 'Grilled pork (Mhoo Ping)',
    21: 'Mango Sticky Rice (Khao Niew Ma Muang)',
    22: 'Minced Pork Omelette (Kai Jeow Moo Saap)',
    23: 'Noodle with chicken curry (Kao Soi)',
    24: 'Pad Thai',
    25: 'Papaya Salad (Somtam)',
    26: 'Pork Satay with Peanut Sauce',
    27: 'Rice Noodles with Fish Curry Sauce (Kanom Jeen Nam Ya)',
    28: 'Shrimp Paste Fried Rice (Khao Kluk Kaphi)',
    29: 'Sour Soup with Shrimp Mixed Veggies (Kaeng Som Kung)',
    30: 'Spicy chicken curry in coconut milk (Tom Kha Gai)',
    31: 'Spicy minced pork salad(Larb mhoo)',
    32: 'Steamed fish with curry paste (Hor mok phra)',
    33: 'Stewed pork legs with rice (Khao Kha Mhoo)',
    34: 'Stir Fried Long Eggplant with Pork (Makheua Yao Pad mhoo)',
    35: 'Stir Fried Pork with Holy Basil and Chilies (Pad ka praow)',
    36: 'Stir fried chicken with cashew nuts',
    37: 'Stir fried clams with roasted chili paste (Hoyrai pad phrik pao)',
    38: 'Stir fried rice noodles with chicken (Kuaitiao khua kai)',
    39: 'Stir-Fried Beef with Yellow Curry Paste (KuaKling)',
    40: 'Stir-Fried Pumpkin with Eggs (Faktong Pad Kai)',
    41: 'Stir-fried crab with curry powder (BooPadPongali)',
    42: 'Stir-fried morning glory with red fire (Pak Boong Fai Dang)',
    43: 'Stuffed bitter gourd broth (Gaeng Jued Mara)',
    44: 'Tapioca balls with pork filling (Sakhu sai mu)',
    45: 'Thai Baked Shrimp with Glass Noodle (Goong Ob Woon Sen)',
    46: 'Thai Massaman Curry Chicken (Massaman Gai)',
    47: 'Thai layer dessert (Khanom Chan)',
    48: 'Thai spicy and sour soup (Tom yum Goong)',
    49: 'Yentafo'
}

datadict_th = {
'Banana in coconut milk (Kluay Buat Chi)': "กล้วยบวชชี",
'Boat Noodles with pork blood (Kuay Teow Reua)': "ก๋วยเตี๋ยวเรือ",
'Cabbage with Fish Sauce (Galam Plee Pad Nam Pla)': "กะหล่ำปลีผัดน้ำปลา",
'Chicken Biryani (Khao Mok Gai)': "ข้าวหมกไก่",
'Chicken Green Curry (Gaeng Kiew Wan)': "แกงเขียวหวาน",
'Chicken Panang Curry (Panang Gai)': "พะแนงไก่",
'Coconut Rlce pancake (Kanom Krok)': "ขนมครก",
'Crispy pork with Chinese broccoli and oyster sauce (Kana Mhoo Krob)': "คะน้าหมูกรอบ",
'Egg and pork in sweet brown sauce (Moo Palo)': "หมูพะโล้",
'Egg custard in pumpkin (Sankaya Faktong)': "สังขยาฟักทอง",
'Fired Rice (Khao Pad)': "ข้าวผัด",
"Fried Boiled Egg with Tamarind Sauce (Son-in-law_s Eggs)": "ไข่ลูกเขย",
'Fried Mussel Pancakes (Hoi Tod)': "หอยทอด",
'Fried fish-paste balls (Tod Mun)': "ทอดมัน",
'Fried noodles in soy sauce (Pad see ew)': "ผัดซีอิ๊ว",
'Garlic and Pepper Pork with Rice (Khao Mhoo Tod Gratiem)': "ข้าวหมูทอดกระเทียม",
'Glass noodle salad (Yam Woon Sen)': "ยำวุ้นเส้น",
'Golden egg yolk threads (Foi Thong)': "ฝอยทอง",
'Grass jelly (Chao Kuay)': "เฉาก๊วย",
'Grilled River Prawns (Goong Phao)': "กุ้งเผา",
'Grilled pork (Mhoo Ping)': "หมูปิ้ง",
'Mango Sticky Rice (Khao Niew Ma Muang)': "ข้าวเหนียวมะม่วง",
'Minced Pork Omelette (Kai Jeow Moo Saap)': "ไข่เจียวหมูสับ",
'Noodle with chicken curry (Kao Soi)': "ข้าวซอย",
'Pad Thai': "ผัดไทย",
'Papaya Salad (Somtam)': "ส้มตำ",
'Pork Satay with Peanut Sauce': "หมูสะเต๊ะ",
'Rice Noodles with Fish Curry Sauce (Kanom Jeen Nam Ya)': "ขนมจีนน้ำยา",
'Shrimp Paste Fried Rice (Khao Kluk Kaphi)': "ข้าวคลุกกะปิ",
'Sour Soup with Shrimp Mixed Veggies (Kaeng Som Kung)': "แกงส้มกุ้ง",
'Spicy chicken curry in coconut milk (Tom Kha Gai)': "ต้มข่าไก่",
'Spicy minced pork salad(Larb mhoo)': "ลาบหมู",
'Steamed fish with curry paste (Hor mok phra)': "ห่อหมกปลา",
'Stewed pork legs with rice(Khao Kha Mhoo)': "ข้าวขาหมู",
'Stir Fried Long Eggplant with Pork (Makheua Yao Pad mhoo)': "มะเขือยาวผัดหมู",
'Stir Fried Pork with Holy Basil and Chilies (Pad ka praow)': "ผัดกะเพรา",
'Stir fried chicken with cashew nuts': "ไก่ผัดเม็ดมะม่วง",
'Stir fried clams with roasted chili paste (Hoyrai pad phrik pao)': "หอยลอยผัดพริกเผา",
'Stir fried rice noodles with chicken (Kuaitiao khua kai)': "ก๋วยเตี๋ยวคั่วไก่",
'Stir-Fried Beef with Yellow Curry Paste(KuaKling)': "คั่วกลิ้ง",
'Stir-Fried Pumpkin with Eggs ( Faktong Pad Kai)': "ฟักทองผัดไข่",
'Stir-fried crab with curry powder(BooPadPongali)': "ปูผัดผงกระหรี่",
'Stir-fried morning glory with red fire (Pak Boong Fai Dang)': "ผักบุ้งไฟแดง",
'Stuffed bitter gourd broth (Gaeng Jued Mara)': "แกงจืดมะระ",
'Tapioca balls with pork filling (Sakhu sai mu)': "สาคูหมู",
'Thai Baked Shrimp with Glass Noodle (Goong Ob Woon Sen)': "กุ้งอบวุ้นเส้น",
'Thai Massaman Curry Chicken (Massaman Gai)': "มัสมั่นไก่",
'Thai layer dessert(Khanom Chan)': "ชนมชั้น",
'Thai spicy and sour soup(Tom yum Goong)': "ต้มยำกุ้ง",
'Yentafo': "เย็นตาโฟ"
}

#rich menu
rich_menu_endpoint = 'https://api.line.me/v2/bot/richmenu'

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {CHANNEL_ACCESS_TOKEN}',
}

rich_menu = {
  "size": {
    "width": 2500,
    "height": 1686
  },
  "selected": True,
  "name": "Rich Menu 1",
  "chatBarText": "Menu",
  "areas": [
    { 
      "bounds": { 
        "x": 17, 
        "y": 13, 
        "width": 1674, 
        "height": 1673 
      }, 
      "action": { 
        "type": "uri", 
        "uri": "https://aroi-thaifood.onrender.com/" 
      } 
    }, 
    { 
      "bounds": { 
        "x": 1780, 
        "y": 51, 
        "width": 695, 
        "height": 754 
      }, 
      "action": { 
        "type": "postback", 
        "data": "changeLanguage" 
      } 
    }, 
    { 
      "bounds": { 
        "x": 1788, 
        "y": 903, 
        "width": 687, 
        "height": 724 
      }, 
      "action": { 
        "type": "postback", 
        "data": "requestInstruction" 
      } 
    } 
  ] 
} 

query_collection = db.collection('user_collection').stream() 
old_user = [] 
for i in query_collection: 
    user_dict = i.to_dict() 
    old_user.append(user_dict["user_id"]) 

try:
    response = requests.post(rich_menu_endpoint, headers=headers, data=json.dumps(rich_menu))
    rich_menu_id = response.json().get('richMenuId')
    rich_menu_image_endpoint = f'https://api.line.me/v2/bot/richmenu/{rich_menu_id}/content'
    headers = {
        'Content-Type': 'image/jpeg',
        'Authorization': f'Bearer {CHANNEL_ACCESS_TOKEN}',
    }
    line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
    try:
        image = open('linerichmenu_2.png', 'rb').read()
        line_bot_api.set_rich_menu_image(rich_menu_id=rich_menu_id, content_type='image/jpeg', content=image)
        line_bot_api.set_default_rich_menu(rich_menu_id)
        if len(old_user) != 0:
            for j in old_user:
                try:
                    line_bot_api.link_rich_menu_to_user(j, rich_menu_id) 
                except Exception as e:
                    app.logger.error('Error:',e) 
    except Exception as e:
        print('Error Image:', e)
except Exception as e :
    print('Error:', e)

#add rich menu to admin
admin_user_id = os.getenv("ADMIN_USER_ID")
line_bot_api.link_rich_menu_to_user(admin_user_id, rich_menu_id)

#process_image
def process_image(image):
    try:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)
        input_shape = (224, 224)  # Example input shape
        image_tensor = tf.image.resize(img, input_shape)
        img_array = np.expand_dims(image_tensor, axis=0)
        return img_array
    
    except Exception as e:
        print(f"An error occurred from process_image function: {e}")
        return None
#prediction   
def prediction(image_path):
    try:
        # บันทึกรูปภาพที่ได้รับจากไลน์
        image_data = process_image(image_path)
        prediction = model.predict(image_data, use_multiprocessing=True)
        predicted_class = tf.argmax(prediction, axis=1)[0]
        predicted_class_name = datadict[int(predicted_class)]
        confidence = tf.reduce_max(prediction)
        confidence_percentage = int(confidence * 100)
        print('predicted_class', predicted_class)
        print("predicted_class_name",predicted_class_name)
        return {"predicted_class_name": predicted_class_name, "confidence_percentage": confidence_percentage}
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return {"error_msg": "An error occurred during prediction."}
def gemini_res(food_name, user):
    genai.configure(api_key=api_key)
    generative_model = genai.GenerativeModel('gemini-pro')
    promt_short_description_en = f'Please provide a {food_name} with the following details answer in english language: Name of the dish , List of ingredients with amounts , Step-by-step cooking method/instructions , Level of spiciness (mild, medium, hot, etc.)'
    promt_short_description_th = f'Please provide a {food_name} with the following details answer in thai language: Name of the dish , List of ingredients with amounts , Step-by-step cooking method/instructions , Level of spiciness (mild, medium, hot, etc.)'
    try:
        if user['language'] == 'en':
            response = generative_model.generate_content(promt_short_description_en)
        else:
            response = generative_model.generate_content(promt_short_description_th)
        return response.text
    except Exception as e:
        print('error from gemini:',e)
        return
    
def gemini_res2(food_name, user):
    genai.configure(api_key=api_key)
    generative_model = genai.GenerativeModel('gemini-pro')
    promt_short_description_en = f""""
            Act as a Thai chef
            Please provide a traditional Thai {food_name} recipe with the following 
            details answer in english language:
            - Dish Name
            - Ingredients
            List ingredients with amounts/quantity
            - Instructions
            Step-by-step cooking method
            - Spice Level 🌶️
            (mild, medium, hot, etc.) with chilli emoji
            - Nutrition 🍴
            Approximate calories, protein, fat, carbs per serving
            - Dietary Notes 📝
            Note any vegetarian, gluten-free, nut-free, etc. options
            - Tips (optional) 💡
            **Please note:** If you don't recognize the {food_name} item as a traditional Thai dish, simply respond with:
            'I'm sorry, I don't recognize "{food_name}" as a traditional Thai dish. Please provide a valid Thai menu item.'.
            Please provide the response in markdown format with Dish Name using # and other topics using ## and emoji to decorate each the topics.
             """
    promt_short_description_th = f""""
            Act as a Thai chef
            Please provide a traditional Thai {food_name} recipe with the following 
            details answer in thai language:
            - Dish Name
            - Ingredients
            List ingredients with amounts/quantity
            - Instructions
            Step-by-step cooking method
            - Spice Level 🌶️
            (mild, medium, hot, etc.) with chilli emoji
            - Nutrition 🍴
            Approximate calories, protein, fat, carbs per serving
            - Dietary Notes 📝
            Note any vegetarian, gluten-free, nut-free, etc. options
            - Tips (optional) 💡
            **Please note:** If you don't recognize the {food_name} item as a traditional Thai dish, simply respond with:
            'I'm sorry, I don't recognize "{food_name}" as a traditional Thai dish. Please provide a valid Thai menu item.'.
            Please provide the response in markdown format with Dish Name using # and other topics using ## and emoji to decorate each the topics.
             """
    try:
        if user['language'] == 'en':
            response = generative_model.generate_content(promt_short_description_en)
        else:
            response = generative_model.generate_content(promt_short_description_th)
        return response.text
    except Exception as e:
        print('error from gemini:',e)
        return
    
def send_flex(header,body):
    res_flex = {
      "type": "bubble",
      "direction": "ltr",
      "header": {
        "type": "box",
        "layout": "vertical",
        "backgroundColor": "#FE5D26",
        "contents": [
          {
            "type": "text",
            "text": "Header",
            "color": "#FFFFFFFF",
            "align": "center",
            "contents": []
          }
        ]
      },
      "body": {
        "type": "box",
        "layout": "vertical",
        "contents": [
          {
            "type": "text",
            "text": "Body",
            "align": "start",
            "wrap": True,
            "contents": []
          }
        ]
      },
      "footer": {
        "type": "box",
        "layout": "horizontal",
        "contents": [
          {
            "type": "filler"
          }
        ]
      }
    }
    #Food name
    res_flex['header']['contents'][0]['text'] = header
    #response contents
    res_flex['body']['contents'][0]['text'] = body
    return res_flex

def send_greeting(user_profile_name): 
    eng_hi = len(f"$ สวัสดีครับ  คุณ {user_profile_name} ผมเป็นบอทที่มีความรู้เกี่ยวกับอาหารไทยเพียงคุณส่งรูปภาพอาหารหรือชื่อเมนูอาหารมาให้ผม ผมก็สามารถบอกชื่ออาหาร พร้อมกับสูตรให้กับคุณครับ\n") 
    greeting = TextSendMessage( 
        text = f"$ สวัสดีครับ  คุณ {user_profile_name} ผมเป็นบอทที่มีความรู้เกี่ยวกับอาหารไทยเพียงคุณส่งรูปภาพอาหารหรือชื่อเมนูอาหารมาให้ผม ผมก็สามารถบอกชื่ออาหาร พร้อมกับสูตรให้กับคุณครับ\n$ Hello {user_profile_name} my name is Aroi and I know everything about thai food I am ready to help you with everything related to thai food. You just send pictures or type me the name of the food I will give you the details and the recipe.", 
        emojis= 
            [{ 
                "index": 0, 
                "productID":"5ac1bfd5040ab15980c9b435", 
                "emojiID": "229" 
            }, 
            { 
                "index": eng_hi, 
                "productID":"5ac1bfd5040ab15980c9b435", 
                "emojiID": "229" 
            }] 
        ) 
    return greeting 

def send_instruction(language, user_id): 
    flex_instruction = { 
          "type": "bubble", 
          "body": { 
            "type": "box", 
            "layout": "vertical", 
            "contents": [ 
              { 
                "type": "text", 
                "text": "Header", 
                "weight": "bold", 
                "size": "sm", 
                "color": "#FFFFFFFF", 
                "wrap":True 
              } 
            ], 
            "backgroundColor": "#FE5D26" 
          }, 
          "footer": { 
            "type": "box", 
            "layout": "vertical", 
            "spacing": "sm", 
            "contents": [ 
              { 
                "type": "text", 
                "text": "Instruction 1", 
                "wrap": True 
              }, 
              { 
                "type": "separator" 
              }, 
              { 
                "type": "text", 
                "text": "Instruction 2", 
                "wrap": True 
              }, 
              { 
                "type": "separator" 
              }, 
              { 
                "type": "text", 
                "text": "Instruction 3", 
                "wrap": True 
              }, 
              { 
                "type": "box", 
                "layout": "vertical", 
                "contents": [], 
                "margin": "sm" 
              } 
            ], 
            "flex": 0 
          }, 
          "styles": { 
            "header": { 
              "backgroundColor": "#FE5D26" 
            } 
          } 
        } 
    if language == 'th': 
        flex_instruction['body']['contents'][0]['text'] = "แนะนำวิธีการใช้งาน บอท Aroi ใช้งานง่ายเพียง 3 ขั้นตอน" 
        flex_instruction['footer']['contents'][0]['text'] = "1.ให้ทำการเลือกรูปภาพอาหารไทยหรือชื่ออาหาร แล้วทำการส่งเข้ามาให้กับเรา" 
        flex_instruction['footer']['contents'][2]['text'] = "2.รอการประมวลผลของรูปภาพอาหารหรือชื่ออาหารที่คุณส่งมาสักครู่" 
        flex_instruction['footer']['contents'][4]['text'] = "3.ทางบอท Aroi ได้ทำการตอบกลับท่านด้วย ชื่ออาหารพร้อมสูตรอาหารให้กับท่าน" 
        line_bot_api.push_message(user_id, FlexSendMessage(alt_text="วิธีการใช้งาน", contents=flex_instruction)) 
        line_bot_api.push_message(user_id, ImageSendMessage(INSTRUCTION_RES,INSTRUCTION_RES)) 
    else: 
        flex_instruction['body']['contents'][0]['text'] = "Instructions on how to use the Aroi bot it’s only take 3 easy steps" 
        flex_instruction['footer']['contents'][0]['text'] = "1.Send me a picture of Thai food or food name." 
        flex_instruction['footer']['contents'][2]['text'] = "2.Wait for Aroi bot to process." 
        flex_instruction['footer']['contents'][4]['text'] = "3.Aroi bot will reply with the name and the recipe for you." 
        line_bot_api.push_message(user_id, FlexSendMessage(alt_text="Instructions", contents=flex_instruction)) 
        line_bot_api.push_message(user_id, ImageSendMessage(INSTRUCTION_RES,INSTRUCTION_RES))          



@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(FollowEvent)
def handle_add_friend(event):
    user_id = event.source.user_id
    profile = line_bot_api.get_profile(user_id).display_name
    language = line_bot_api.get_profile(user_id).language
    user = {
        'user_id':user_id,
        'language':language,
        'history':[]
        }
    db.collection('user_collection').document(user_id).set(user)

    #Link rich menu to user
    try:
        user_rich_menu = line_bot_api.get_rich_menu_id_of_user(user_id)
        if rich_menu_id != user_rich_menu['richMenuId']:
            if rich_menu_id:
                line_bot_api.link_rich_menu_to_user(user_id, rich_menu_id)
    except:
        line_bot_api.link_rich_menu_to_user(user_id, rich_menu_id)
    line_bot_api.push_message(user_id, send_greeting(profile))
    line_bot_api.push_message(user_id, ImageSendMessage(MENU_GUIDE, MENU_GUIDE))

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    message_text = event.message.text
    user_id = event.source.user_id
    profile = line_bot_api.get_profile(user_id)
    display_name = profile.display_name
    current_date_time = datetime.now()

    # Format the date as a string
    current_date_string = current_date_time.strftime("%Y-%m-%d")
    user = db.collection('user_collection').where('user_id', '==', user_id).stream()
    user = [i.to_dict() for i in user]
    if len(user) == 0:
        language = profile.language
        user = {
            'user_id':user_id,
            'language':language,
            'history':[]
            }
        db.collection('user_collection').document(user_id).set(user)
        user = db.collection('user_collection').where('user_id', '==', user_id).stream()
        user = [i.to_dict() for i in user]
    user = user[0]
    language = user['language']
    
    print(f"Received message '{message_text}' from user '{display_name}'")
    if language != 'th':
       res_text = 'You are looking for'
       alt ='Recipe'
       flex_bar = 'Something went wrong'
    else:
       res_text = 'คุณกำลังค้นหาสูตร'
       alt = 'สูตรอาหาร' 
       flex_bar = 'เกิดข้อผิดพลาดบางอย่าง'

    if message_text:
        print(f"Users'{display_name}' send '{message_text}'")
        res = gemini_res2(message_text, user)
        if len(res)>200 : 
        #firestore: update data
          user['history'].append({'date':current_date_string, 'dish':message_text})
          db.collection('user_collection').document(user_id).update(user)
          flex_change = send_flex(f"{message_text}", res)
          flex_content = FlexSendMessage(alt_text=alt, contents = flex_change)
          line_bot_api.reply_message(event.reply_token, TextSendMessage(text= f"{res_text} {message_text}"))
          line_bot_api.push_message(
              user_id,
              flex_content
          )
        else : 
          user['history'].append({'date':current_date_string, 'dish':message_text})
          db.collection('user_collection').document(user_id).update(user)
          flex_change = send_flex(f"{flex_bar}", res)
          flex_content = FlexSendMessage(alt_text=alt, contents = flex_change)
          line_bot_api.reply_message(event.reply_token, TextSendMessage(text= f"{res_text} {message_text}"))
          line_bot_api.push_message(
              user_id,
              flex_content
          )

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_id= event.message.id
    message_content = line_bot_api.get_message_content(message_id)
    user_id = event.source.user_id
    profile = line_bot_api.get_profile(user_id)
    display_name = profile.display_name
    current_date_time = datetime.now()
    # Format the date as a string
    current_date_string = current_date_time.strftime("%Y-%m-%d")
    user = db.collection('user_collection').where('user_id', '==', user_id).stream()
    user = [i.to_dict() for i in user]
    if len(user) == 0:
        language = profile.language
        user = {
            'user_id':user_id,
            'language':language,
            'history':[]
            }
        db.collection('user_collection').document(user_id).set(user)
        user = db.collection('user_collection').where('user_id', '==', user_id).stream()
        user = [i.to_dict() for i in user]
    user = user[0]
    language = user['language']

    print(f"Received image from user '{display_name}'")
    if language != 'th':
       res_text = 'that is'
       error_text = "Sorry, I can't recognize this dish"
       alt ='Recipe'
    else:
        res_text = 'อาหารที่คุณส่งมาคือ'
        error_text = 'ขออภัย รายการนี้ยังไม่พร้อมใช้งาน'
        alt = 'สูตรอาหาร'
    try:
        with open("pic/image.jpg", 'wb') as f:
            for chunk in message_content.iter_content():
                f.write(chunk)
        
        # Process the image here (e.g., use a machine learning model)
        prediction_result = prediction("pic/image.jpg")
        
        res = gemini_res(prediction_result["predicted_class_name"], user)
        flex_change = send_flex(f"{datadict_th[prediction_result['predicted_class_name']]} ({prediction_result['predicted_class_name']})", res)
        flex_content = FlexSendMessage(alt_text=alt, contents = flex_change)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text= f"{res_text} {datadict_th[prediction_result['predicted_class_name']]} ({prediction_result['predicted_class_name']})"))
        line_bot_api.push_message(
            user_id,
            flex_content
        )
        #firestore: update data
        user['history'].append({'date':current_date_string, 'dish':prediction_result["predicted_class_name"]})
        db.collection('user_collection').document(user_id).update(user)
    except LineBotApiError as e:
        line_bot_api.push_message(user_id, TextSendMessage(text= error_text))
        app.logger.exception(e)
        abort(500)

@handler.add(PostbackEvent)
def handler_postback_option(event): 
    user_id = event.source.user_id 
    postback = event.postback.data 
    user = db.collection('user_collection').where('user_id', '==', user_id).stream() 
    user = [i.to_dict() for i in user] 
    user = user[0] 
    profile = line_bot_api.get_profile(user_id) 
    language =user['language'] 
    if postback == 'changeLanguage': 
        if language == 'en': 
            reply_message = 'คุณได้เปลี่ยนภาษาเป็นภาษาไทย' 
            language = 'th' 
        else: 
            reply_message = 'You have changed your language preference to English' 
            language = 'en' 
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text= reply_message)) 

        #firestore: update data 
        db.collection('user_collection').document(user_id).update({'language':language}) 
    if postback == 'requestInstruction': 
        if language == 'th': 
            send_instruction(language, user_id) 
        else: 
            send_instruction(language, user_id) 
        
if __name__ == "__main__":
    app.run()