module test_circuit (input A, input B, input C, output Y, output Z);
  wire w1;
  and g1(w1, A, B);
  or g2(Y, w1, C);
  xor g3(Z, A, B);
endmodule
