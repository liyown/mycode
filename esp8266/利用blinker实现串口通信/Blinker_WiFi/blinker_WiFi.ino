
#define BLINKER_WIFI

#include <Blinker.h>//包含点灯科技的库文件

char auth[] = "25c40dc37c29";//点灯科技上设备名字
char ssid[] = "SXLMm";//要连接的wifi
char pswd[] = "12345678";//密码

// 新建组件对象
BlinkerButton Button1("btn-qj");
BlinkerButton Button2("btn-ht");
BlinkerButton Button3("btn-zz");
BlinkerButton Button4("btn-yz");
BlinkerButton Button5("btn-tz");
void button1_callback(const String & state)//前进，按钮回调函数
{
    Serial.print("Q");
}
void button2_callback(const String & state)//后退
{
    Serial.print("H");
}
void button3_callback(const String & state)//左转
{
    Serial.print("Z");
}
void button4_callback(const String & state)//右转
{
    Serial.print("Y");
}
void button5_callback(const String & state)//停止
{
    Serial.print("T");
}
void setup()
{

    Serial.begin(115200);//打开串口通信
    //初始化
    pinMode(LED_BUILTIN, OUTPUT);//设置自带led灯输出
    Blinker.begin(auth, ssid, pswd);开始连接wifi
    Serial.println("");
    Serial.print("connecting to  ");
    Serial.print(ssid);
    /* 检验WiFi是否连接 */
    while(WiFi.status()!=WL_CONNECTED)//检验WiFi是否连接，其中WiFi.ststus连接好输出WL_CONNECTED
    {
      delay(1000);
      digitalWrite(LED_BUILTIN,!digitalRead(LED_BUILTIN));
      Serial.print(".");
    }
    Serial.println("succesly  connected");
   
    //绑定回调函数
    Button1.attach(button1_callback);
    Button2.attach(button2_callback);
    Button3.attach(button3_callback);
    Button4.attach(button4_callback);
    Button5.attach(button5_callback);
}

void loop() {
    Blinker.run();//运行总函数
}
