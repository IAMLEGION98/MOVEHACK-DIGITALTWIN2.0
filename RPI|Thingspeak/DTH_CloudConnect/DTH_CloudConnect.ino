#include <dht.h>
#include <Bridge.h>
#include <HttpClient.h>
#include <WiFi101.h>

dht DHT;
int sensorPin = 6;
WiFiClient client;
char thingSpeakAddress[] = "api.thingspeak.com";
String APIKey = "EIIV30W3IB5MCIS0";             // enter your channel's Write API Key
const int updateThingSpeakInterval = 20 * 1000; 


#define DHT11_PIN 7

void setup(){
  Serial.begin(9600);
}

void loop()
{
  
  int chk = DHT.read11(DHT11_PIN);
  Serial.print("Temperature = ");
  Serial.println(DHT.temperature);
  Serial.print("Humidity = ");
  Serial.println(DHT.humidity);  
  HttpClient client;


  float sensorValue = digitalRead(sensorPin);
  Serial.print("Smake = ");
  Serial.println(sensorValue);
  
  

  
  delay(1000);
}
void updateThingSpeak(String tsData) {
  if (client.connect(thingSpeakAddress, 80)) {
    client.print("POST /update HTTP/1.1\n");
    client.print("Host: api.thingspeak.com\n");
    client.print("Connection: close\n");
    client.print("X-THINGSPEAKAPIKEY: " + APIKey + "\n");
    client.print("Content-Type: application/x-www-form-urlencoded\n");
    client.print("Content-Length: ");
    client.print(tsData.length());
    client.print("\n\n");
    client.print(tsData);
    

    if (client.connected()) {
      Serial.println("Connecting to ThingSpeak...");
      Serial.println();
    }
  }
}


