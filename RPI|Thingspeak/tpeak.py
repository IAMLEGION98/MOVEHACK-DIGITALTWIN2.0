#!/usr/bin/python
import sys
import Adafruit_DHT

import httplib, urllib
import time
sleep = 10 # how many seconds to sleep between posts to the channel
key = 'EIIV30W3IB5MCIS0'  # Thingspeak channel to update


#Report Raspberry Pi internal temperature to Thingspeak Channel
def thermometer():
    while True:
        #Calculate CPU temperature of Raspberry Pi in Degrees C
       # temp = int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1e3 # Get Raspberry Pi CPU temp
           humidity, temperature = Adafruit_DHT.read_retry(11, 4)
           TWF = 9/5*temperature+32
        params = urllib.urlencode({'field1': temperature,'field2':TWF,'field3':humidity,'key':key }) 
        headers = {"Content-typZZe": "application/x-www-form-urlencoded","Accept": "text/plain"}
        conn = httplib.HTTPConnection("api.thingspeak.com:80")
        try:
            conn.request("POST", "/update", params, headers)
            response = conn.getresponse()
            print temperature
            print response.status, response.reason
            data = response.read()
            conn.close()
        except:
            print "connection failed"
        break
#sleep for desired amount of time
if __name__ == "__main__":
        while True:
                thermometer()
                time.sleep(sleep)

