module fsm_ck11_2(out,clk,rst,T);
    input  clk,rst,T;
	output out;
	reg out;
	reg[1:0] state,next_state;
	parameter S0=2'b00,S1=2'b01,S2=2'b10;
	always @(posedge clk,posedge rst)
	begin
		if(rst) state=S0;
		else state=next_state;
	end
	always @(state,T)
	begin 
	  case(state)
		S0:
		   begin
			out=1'b0;
			if(T) next_state=S1;
			else  next_state=S0;
		   end
		S1:
		   begin
			out=1'b0;
			if(T) next_state=S2;
			else  next_state=S0;
		   end
		S2:
		    begin
			out<=1'b1;
			if(T) next_state=S2;
			else  next_state=S0;
		    end
		default:
		  begin 
			next_state=S0;
			out=1'b0;
		  end
		endcase
	end
endmodule
