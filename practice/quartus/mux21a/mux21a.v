module mux21a(a,b,s,y);
input a,b,s;
output y;
reg y;
always @(a or b or s)
begin
if(s==1'b0)
begin
y<=a;
end
else
begin
y<=b;
end
end
endmodule
