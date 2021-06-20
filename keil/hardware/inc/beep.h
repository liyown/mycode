#ifndef _BEEP_H_
#define _BEEP_H_

#include "sys.h" 
typedef struct
{

	_Bool Beep_Status;

} BEEP_INFO;

#define BEEP_ON		1

#define BEEP_OFF	0

#define BEEP PBout(8)	// BEEP,·äÃùÆ÷½Ó¿Ú	
extern BEEP_INFO beep_info;

void BEEP_Init(void);
//void Beep_Init(void);

//void Beep_Set(_Bool status);


#endif
