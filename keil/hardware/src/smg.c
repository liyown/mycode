#include "smg.h"

void LED_SMG_Init(void)
{

   RCC->APB2ENR|=1<<3;    //使能PORTB时钟	 
   RCC->APB2ENR|=1<<4;    //使能PORTC时钟
   RCC->APB2ENR|=1<<0;    //使能AFIO时钟
   
   JTAG_Set(SWD_ENABLE);//关闭jtag，使能SWD，可以用SWD模式调试 PB4做普通IO口使用,否则不能当IO使用
   
   GPIOB->CRL&=0XFF000FFF; 
   GPIOB->CRL|=0X00333000;//PB3~5推挽输出
    
   GPIOC->CRH&=0XFFF000FF; 
   GPIOC->CRH|=0X00033300;//PC10~12推挽输出  
    	
   GPIOB->BRR = 1<<3|1<<5;
   GPIOB->BSRR = 1<<4;
   GPIOC->BSRR = 1<<10|1<<11|1<<12;
   
}
//74HC138驱动
//数码管位选
//num:要显示的数码管编号 0-7(共8个数码管)
void LED_Wei(u8 num)
{
    LED_A0=num&0x01;
	LED_A1=(num&0x02)>>1;
	LED_A2=(num&0x04)>>2;
}
//74HC595驱动
//数码管显示
//duan:显示的段码
//wei:要显示的数码管编号 0-7(共8个数码管)
void LED_Write_Data(u8 duan,u8 wei)
{
	u8 i;
	for( i=0;i<8;i++)//先送段
	{
		LED_DS=(duan>>i)&0x01;
		LED_SCK=0;
		delay_us(5);
		LED_SCK=1;
	}
    LED_Wei(wei);//后选中位
	LED_Refresh();
}
//74HC595驱动
//数码管刷新显示
void LED_Refresh(void)
{
	LED_LCLK=1;
	delay_us(5);
	LED_LCLK=0;
}
