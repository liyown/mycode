#include "INTR.h"
#include "key.h"
#include "delay.h"
#include "led.h"
#include "beep.h"
#include "esp8266.h"
#include "onenet.h"
//�ⲿ�жϳ�ʼ������
int flag = 1;
int time = 500;
void delay_nms(u16 time)
{    
   u16 i=0;  
   while(time--)
   {
      i=12000;  //�Լ�����
      while(i--) ;    
   }
}
void EXTIX_Init(void)
{
 
    //KEY_Init();//��ʼ��������Ӧioģʽ
//    Ex_NVIC_Config(GPIO_C,8,FTIR); 		//�½��ش���
//	Ex_NVIC_Config(GPIO_C,9,FTIR);		//�½��ش���
	Ex_NVIC_Config(GPIO_D,2,FTIR);		//�½��ش���

	//MY_NVIC_Init(2,1,EXTI9_5_IRQn,2);  	//��ռ2�������ȼ�1����2
	MY_NVIC_Init(2,0,EXTI2_IRQn,2);  	//��ռ2�������ȼ�0����2	  
 
} 
//�ⲿ�ж�2�������
extern _Bool connectstatus;
void EXTI2_IRQHandler(void)
{
    delay_nms(10);    //����			 
    if(KEY2==0)	     //KEY2����
    {
		PCout(7)=0;
		PCout(0)=1;
		PCout(1)=1;
		ESP8266_Init();					//��ʼ��ESP8266
		while(OneNet_DevLink())			//����OneNET
		{
			delay_ms(500);
		}
		if(connectstatus == 0){PCout(0)=0;	BEEP = 0;}				//������ʾ����ɹ�
		else PCout(1)=0;
		PCout(7)=1;
    }
    EXTI->PR=1<<2;     //���LINE2�ϵ��жϱ�־λ
}
//�ⲿ�ж�5_9�����жϳ���
 void EXTI9_5_IRQHandler(void)
{		
	delay_nms(10);   //����			 
	if(KEY0==0)	    //KEY0����
	{
		
	}
 	if(KEY1==0)	    //KEY1����
	{

	}
	EXTI->PR=1<<8;  //���LINE8�ϵ��жϱ�־λ  
	EXTI->PR=1<<9;  //���LINE9�ϵ��жϱ�־λ  	
}

