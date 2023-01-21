#include "pwm.h"

//TIM3 PWM�����ʼ��
//arr���Զ���װֵ
//psc��ʱ��Ԥ��Ƶ��
void TIM3_PWM_Init(u16 arr,u16 psc)
{  

	RCC->APB1ENR|=1<<1;	//TIM3ʱ��ʹ�� 
	RCC->APB2ENR|=1<<2; //ʹ��PORTCʱ��	 
	GPIOA->CRL&=0XF0FFFFFF;	//PC6���
	GPIOA->CRL|=0X0B000000;	//���ù������ 	  
	RCC->APB2ENR|=1<<0;     //ʹ��AFIOʱ��	
    AFIO->MAPR&=0XFFFFF3FF; //���MAPR��[11:10]
	AFIO->MAPR|=0<<10;      //������ӳ��,TIM3_CH1->PC6
	TIM3->ARR=arr;			//�趨�������Զ���װֵ 
	TIM3->PSC=psc;			//Ԥ��Ƶ������Ƶ
	TIM3->CCMR1|=7<<4;  	//CH1 PWM2ģʽ		 
	TIM3->CCMR1|=1<<3; 	    //CH1Ԥװ��ʹ��	   
	TIM3->CCER|=1<<0;   	//OC1 ���ʹ��	   
	TIM3->CR1=0x0080;   	//ARPEʹ�� 
	TIM3->CR1|=0x01;    	//ʹ�ܶ�ʱ��3 	
		
}

