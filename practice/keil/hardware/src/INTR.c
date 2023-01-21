#include "INTR.h"
#include "key.h"
#include "delay.h"
#include "led.h"
#include "beep.h"
#include "esp8266.h"
#include "onenet.h"
//外部中断初始化函数
int flag = 1;
int time = 500;
void delay_nms(u16 time)
{    
   u16 i=0;  
   while(time--)
   {
      i=12000;  //自己定义
      while(i--) ;    
   }
}
void EXTIX_Init(void)
{
 
    //KEY_Init();//初始化按键对应io模式
//    Ex_NVIC_Config(GPIO_C,8,FTIR); 		//下降沿触发
//	Ex_NVIC_Config(GPIO_C,9,FTIR);		//下降沿触发
	Ex_NVIC_Config(GPIO_D,2,FTIR);		//下降沿触发

	//MY_NVIC_Init(2,1,EXTI9_5_IRQn,2);  	//抢占2，子优先级1，组2
	MY_NVIC_Init(2,0,EXTI2_IRQn,2);  	//抢占2，子优先级0，组2	  
 
} 
//外部中断2服务程序
extern _Bool connectstatus;
void EXTI2_IRQHandler(void)
{
    delay_nms(10);    //消抖			 
    if(KEY2==0)	     //KEY2按键
    {
		PCout(7)=0;
		PCout(0)=1;
		PCout(1)=1;
		ESP8266_Init();					//初始化ESP8266
		while(OneNet_DevLink())			//接入OneNET
		{
			delay_ms(500);
		}
		if(connectstatus == 0){PCout(0)=0;	BEEP = 0;}				//鸣叫提示接入成功
		else PCout(1)=0;
		PCout(7)=1;
    }
    EXTI->PR=1<<2;     //清除LINE2上的中断标志位
}
//外部中断5_9服务中断程序
 void EXTI9_5_IRQHandler(void)
{		
	delay_nms(10);   //消抖			 
	if(KEY0==0)	    //KEY0按键
	{
		
	}
 	if(KEY1==0)	    //KEY1按键
	{

	}
	EXTI->PR=1<<8;  //清除LINE8上的中断标志位  
	EXTI->PR=1<<9;  //清除LINE9上的中断标志位  	
}

