module fsm_ck11(output reg out,input clk,rst,x);
	reg[1:0] state,next_state;
	parameter S0=1'b0,S1=1'b1;
	always @(posedge clk or posedge rst)
	begin 
		if(rst) state<=S0;
		else    state<=next_state;
	end
	always @(state or x)
	begin case(state)
		  S0: 
			begin
				if(x) begin next_state<=S1;out<=1'b0;end
				else  begin next_state<=S0;out<=1'b0;end
			end
		  S1:
			begin
				if(x) begin next_state<=S1;out<=1'b1;end
				else  begin next_state<=S0;out<=1'b0;end
			end
		  default:begin next_state<=S0;out<=1'b0;end
		  endcase
	end
endmodule
