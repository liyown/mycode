onerror {quit -f}
vlib work
vlog -work work 4bitadder.vo
vlog -work work 4bitadder.vt
vsim -novopt -c -t 1ps -L cycloneii_ver -L altera_ver -L altera_mf_ver -L 220model_ver -L sgate work.4bitadder_vlg_vec_tst
vcd file -direction 4bitadder.msim.vcd
vcd add -internal 4bitadder_vlg_vec_tst/*
vcd add -internal 4bitadder_vlg_vec_tst/i1/*
add wave /*
run -all
