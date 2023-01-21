module fenpin(clkin,clk6hz,clk190hz,clk760hz);
input clkin;
output clk6hz,clk190hz,clk760hz;
reg[25:0] qout;
always @(posedge clkin)
begin
 qout<=qout+1;
end
assign clk6hz=qout[23];
assign clk190hz=qout[18];
assign clk760hz=qout[16];
endmodule