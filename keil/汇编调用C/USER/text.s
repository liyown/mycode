	PRESERVE8
	export XMUL
	AREA XMUL,CODE,READONLY
	import MUL1
	str lr,[sp,#-4]!
	mov r0,#1
	mov r1,#1
	mov r2,#1
	mov r3,#1
	str r0,[sp,#-4]!
	str r0,[sp,#-4]!
	bl MUL1
	add sp,sp,#8
	ldr pc,[sp],#4	
	END

	