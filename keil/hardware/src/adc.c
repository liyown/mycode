#include "adc.h"

void Init_ADC0809(void)//初始化与ADC0809接口的GPIO引脚及功能
{
	GPIO_InitTypeDef GPIO_InitStructure;
//(1)设置PC0—PC7为输入模式	数据输入
	/*开启按键端口的时钟*/
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOC,ENABLE);
	//选择按键的引脚
	GPIO_InitStructure.GPIO_Pin = 0x0ff; 
	// 设置按键的引脚为浮空输入
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING; 
	//使用结构体初始化按键
	GPIO_Init(GPIOC, &GPIO_InitStructure);
//(2)设置PC8为输入模式  EOC结束标志高电平有效
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_8;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING; 
	GPIO_Init(GPIOC, &GPIO_InitStructure);
//(3)设置PBA0-PA5为输出模式		
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA,ENABLE);
	GPIO_InitStructure.GPIO_Pin = 0x3f; 
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP; 
	GPIO_Init(GPIOA, &GPIO_InitStructure);
	
}
void SelectChannel(int ch)//选择ADC模拟输入通道函数
{
//(1)初始化锁存信号状态
	ALE = 0;
//(2)延时
	delay_us(10);
//(3)把通道地址输出到GPIO对应引脚上
	OUT_ADDR(ch);
//(4)延时等待数据稳定
	delay_us(10);
//(5)产生有效地址锁存脉冲
	ALE = 1;delay_us(10);
	ALE = 0;
}
int ReadAdc0809(int ch)//启动采集指定通道ADC变换、等待变换完毕、读取转换结果，返回主函数。
{
//(1)初始化START信号状态
	START = 0;
//(2)调用SelectChannel()函数选择AIN输入通道	
	SelectChannel(ch);
//(3)启动转换
	START = 1;delay_us(10);
	START = 0;
//(4)等待转换结束
	while(EOC==1);
	while(EOC==0);
//(5)读取转换结果
	OE = 1;delay_us(10);
	unsigned int ending = IN_DATA;
	OE = 0;
//(6)将结果返回主函数
	return(ending);
}
