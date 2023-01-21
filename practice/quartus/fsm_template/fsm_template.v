module fsm_template(out,clk,rst,x);
	input clk,rst,x;
	output out;
	reg out;
	reg[1:0] state,next_state;
	parameter S0=2'b00,S1=2'b01,S2=2'b10,S3=2'b11;
	always @(posedge clk,posedge rst)
	begin 
		if(rst) state<=S0;
		else state<=next_state;
	end
	always @(state,x)
	begin 
		case(state)
			S0:begin
				out<=1'b0;
				if(x) next_state<=S1;
				else  next_state<=S0;
			end
			S1:
            begin
				out<=1'b0;
				if(x) next_state<=S1;
				else  next_state<=S2;
			end
			S2: 
			begin
				out<=1'b0;
				if(x) next_state<=S3;
				else  next_state<=S0;
			end
			S3:
			begin
				out<=1'b1;
				if(x) next_state<=S1;
				else  next_state<=S2;
			end
			default :
			begin 
				next_state<=S0;
				out<=1'b0;
			end
		endcase
	end
endmodule 
