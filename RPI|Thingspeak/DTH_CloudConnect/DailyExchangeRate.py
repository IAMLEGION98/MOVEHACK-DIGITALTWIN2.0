import gspread
from oauth2client.service_account import ServiceAccountCredentials
from selenium import webdriver
import time
scope =['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json',scope)
client= gspread.authorize(creds)
sheet = client.open('TradersMusthave').sheet1
from bs4 import BeautifulSoup
driver = webdriver.Chrome()
url= "http://www.xe.com/currencyconverter/convert/?Amount=1&From=USD&To=INR"
driver.maximize_window()
driver.get(url)
traders = sheet.get_all_records()
print(traders)

time.sleep(5)
content = driver.page_source.encode('utf-8').strip()
soup = BeautifulSoup(content,"html.parser")
officials = soup.find("span",{"class":"uccResultAmount"})

pric = officials.text
print(pric)
sheet.update_cell(2,2,pric)



driver.quit()

