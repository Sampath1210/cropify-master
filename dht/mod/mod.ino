#include "DHT.h"

#define DHTPIN 7

#define DHTTYPE DHT11

DHT dht(DHTPIN, DHTTYPE);

int sensorPin = A0; 
int sensorValue;

void setup() {
  Serial.begin(9600);
  dht.begin();
}

void loop() {
  sensorValue = analogRead(sensorPin);
  float hum = dht.readHumidity();
  float temp = dht.readTemperature();
  Serial.print(temp);
  Serial.print("x");
  Serial.print(hum);
  Serial.print("x");
  Serial.println(sensorValue);
  delay(10000);
}
