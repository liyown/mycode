#ifndef __KEY_H
#define	__KEY_H
#include "sys.h"
#include "delay.h"
//  Òý½Å¶¨Òå
#define KEY0	PCin(8)  
#define KEY1	PCin(9)   
#define KEY2	PDin(2) 


#define KEY_ON	1
#define KEY_OFF	0

void Key_GPIO_Config(void);
uint8_t Key_Scan(GPIO_TypeDef* GPIOx,uint16_t GPIO_Pin);
uint8_t KEY_Scan(int mode);

#endif
