module sled(A,LED7S,dig);
input[3:0] A;
output [7:0] LED7S;
output [7:0] dig; //�����ʹ�ܼĴ���
reg [7:0] LED7S; //����ܶ�������Ĵ���
assign dig=8'b11111110;//ʹ�ܵ�һ�������
always @ (A) 
begin 
 case (A) 
 4'h0 : LED7S = 8'hc0; //��ʾ"0" 
 4'h1 : LED7S = 8'b11110101;//��ʾ"1" 
 4'h2 : LED7S = 8'b10100100;//��ʾ"2" 
 4'h3 : LED7S = 8'b10110000;//��ʾ"3" 
 4'h4 : LED7S = 8'b11001001;//��ʾ"4" 
 4'h5 : LED7S = 8'b10010010;//��ʾ"5" 
 4'h6 : LED7S = 8'b10000010;//��ʾ"6" 
 4'h7 : LED7S = 8'b11111000;//��ʾ"7" 
 4'h8 : LED7S = 8'b10000000;//��ʾ"8" 
 4'h9 : LED7S = 8'b10010000;//��ʾ"9" 
 4'ha : LED7S = 8'b10001000;//��ʾ"a" 
 4'hb : LED7S = 8'b10000011;//��ʾ"b" 
 4'hc : LED7S = 8'b11000110;//��ʾ"c" 
 4'hd : LED7S = 8'b10100001;//��ʾ"d" 
 4'he : LED7S = 8'b10001000;//��ʾ"e" 
 4'hf : LED7S = 8'b10001110;//��ʾ"f" 
 default:LED7S =8'h00; 
 endcase 
end 
endmodule
