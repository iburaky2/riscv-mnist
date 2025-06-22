module memory_mux (
  // CPU interface
  input  wire cpu_valid_i,
  output reg  cpu_ready_o,
  input  wire [31:0] cpu_addr_i,
	input  wire [31:0] cpu_wdata_i,
	input  wire [3:0]  cpu_wstrb_i,
	output reg  [31:0] cpu_rdata_o,
  // RAM interface
  output reg  mem_valid_o,
  input  wire mem_ready_i,
  output reg  [31:0] mem_addr_o,
  output reg  [31:0] mem_wdata_o,
  output reg  [3:0]  mem_wstrb_o,
  input  wire [31:0] mem_rdata_i,
  // Camera interface
  output reg  cam_capture_o,
  output reg  cam_read_valid_o,
  input  wire [7:0] cam_pixel_i
);
  // Processes
  always @(*) begin
    if (cpu_valid_i) begin
      case (cpu_addr_i)
        32'h2000_0000: begin 
          if (cpu_wstrb_i != 4'd0) begin
            cpu_ready_o   = 1'b1;
            cam_capture_o = 1'b1;
          end else begin
            cpu_ready_o   = 1'b0;
            cam_capture_o = 1'b0;
          end
        end

        32'h2000_0004: begin
          cpu_ready_o = 1'b1;
          cam_read_valid_o = 1'b1;
          cpu_rdata_o = {24'd0, cam_pixel_i};
        end

        default: begin
          mem_valid_o = cpu_valid_i;
          cpu_ready_o = mem_ready_i;
          mem_addr_o  = cpu_addr_i;
          mem_wdata_o = cpu_wdata_i;
          mem_wstrb_o = cpu_wstrb_i;
          cpu_rdata_o = mem_rdata_i;
          cam_read_valid_o = 1'b0;
          cam_capture_o    = 1'b0;
        end
      endcase
    end else begin
      cpu_ready_o      = 1'b0;
      mem_valid_o      = 1'b0;
      mem_addr_o       = 32'd0;
      mem_wdata_o      = 32'd0;
      mem_wstrb_o      = 32'd0;
      cam_capture_o    = 1'b0;
      cam_read_valid_o = 1'b0;
    end
  end
endmodule
