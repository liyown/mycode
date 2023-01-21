#ifndef 	_SMG_H
#define 	_SMG_H

#include "sys.h"
#include "delay.h"
////74HC138 操作线
#define LED_A0 PCout(10) //A0 地址线
#define LED_A1 PCout(11) //A1 地址线
#define LED_A2 PCout(12) //A2 地址线
////74HC595 操作线
#define LED_DS PBout(3) //数据线
#define LED_LCLK PBout(4) //锁存时钟线
#define LED_SCK PBout(5) //时钟线
void LED_SMG_Init(void);
void LED_Refresh(void);
void LED_Write_Data(u8 duan,u8 wei);
#endif

