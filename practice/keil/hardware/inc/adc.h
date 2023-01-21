#ifndef __ADC__
#define __ADC__
#include "sys.h"
#include "delay.h"

#define START  PAout(4)
#define EOC    PCin(8) 
#define OE     PAout(5)
#define IN_DATA   (GPIOC->IDR&0xff) //��������Ĵ���
#define ALE    PAout(3)  
#define OUT_ADDR(addr)\
        GPIOA->ODR = ((GPIOA->ODR&(~0x7)) | addr)//��ַ���뺯��


void Init_ADC0809(void);
void SelectChannel(int ch);
int ReadAdc0809(int ch);
#endif

