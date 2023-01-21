#include "adc.h"

void Init_ADC0809(void)//��ʼ����ADC0809�ӿڵ�GPIO���ż�����
{
	GPIO_InitTypeDef GPIO_InitStructure;
//(1)����PC0��PC7Ϊ����ģʽ	��������
	/*���������˿ڵ�ʱ��*/
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOC,ENABLE);
	//ѡ�񰴼�������
	GPIO_InitStructure.GPIO_Pin = 0x0ff; 
	// ���ð���������Ϊ��������
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING; 
	//ʹ�ýṹ���ʼ������
	GPIO_Init(GPIOC, &GPIO_InitStructure);
//(2)����PC8Ϊ����ģʽ  EOC������־�ߵ�ƽ��Ч
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_8;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING; 
	GPIO_Init(GPIOC, &GPIO_InitStructure);
//(3)����PBA0-PA5Ϊ���ģʽ		
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA,ENABLE);
	GPIO_InitStructure.GPIO_Pin = 0x3f; 
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP; 
	GPIO_Init(GPIOA, &GPIO_InitStructure);
	
}
void SelectChannel(int ch)//ѡ��ADCģ������ͨ������
{
//(1)��ʼ�������ź�״̬
	ALE = 0;
//(2)��ʱ
	delay_us(10);
//(3)��ͨ����ַ�����GPIO��Ӧ������
	OUT_ADDR(ch);
//(4)��ʱ�ȴ������ȶ�
	delay_us(10);
//(5)������Ч��ַ��������
	ALE = 1;delay_us(10);
	ALE = 0;
}
int ReadAdc0809(int ch)//�����ɼ�ָ��ͨ��ADC�任���ȴ��任��ϡ���ȡת�������������������
{
//(1)��ʼ��START�ź�״̬
	START = 0;
//(2)����SelectChannel()����ѡ��AIN����ͨ��	
	SelectChannel(ch);
//(3)����ת��
	START = 1;delay_us(10);
	START = 0;
//(4)�ȴ�ת������
	while(EOC==1);
	while(EOC==0);
//(5)��ȡת�����
	OE = 1;delay_us(10);
	unsigned int ending = IN_DATA;
	OE = 0;
//(6)���������������
	return(ending);
}
