// Focused behavioral regression for the local TTsim 1.10.6 Blackhole packer model.
// Compile against the actual upstream translation unit; no copied numerical model.
#include "tensix.cpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

ChipState g_chips[NUM_CHIPS];
uint64_t ttsim_get_clock() { return 0; }

static constexpr uint32_t output_addr = 0x1000;
static constexpr uint32_t face_rows = 16;
static constexpr uint32_t pipe_id = 2;

static void check(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(2);
    }
}

static TensixState *configure(uint32_t format, bool dst32, bool disable_flags, bool accumulate) {
    TensixState *s = &g_t_tiles[0].tensix[0];
    tensix_init(s, 0);
    auto &c = s->config[0];
    c.PACK_COUNTERS_SEC0_pack_reads_per_xy_plane = face_rows;
    c.THCON_SEC0_REG1_Disable_zero_compress = 1;
    c.THCON_SEC0_REG1_Disable_pack_zero_flags = disable_flags;
    c.THCON_SEC0_REG1_Pack_L1_Acc = accumulate;
    c.ALU_FORMAT_SPEC_REG2_Dstacc = format;
    c.THCON_SEC0_REG1_In_data_format = format;
    c.THCON_SEC0_REG1_Out_data_format = format;
    c.PCK_DEST_RD_CTRL_Read_32b_data = dst32;
    c.ALU_ACC_CTRL_Fp32_enabled = dst32;
    c.PCK0_ADDR_CTRL_XY_REG_0_Ystride = format == 0 ? 64 : 32;
    c.PCK_EDGE_OFFSET_SEC0_mask = 0xffff;
    c.THCON_SEC0_REG1_L1_Dest_addr = output_addr / 16 - 1;
    s->addr_ctrl[pipe_id][2].ch1_x = 15;
    return s;
}

static void pack_face(TensixState *s) {
    for (uint32_t row = 0; row < face_rows; ++row) {
        s->addr_ctrl[pipe_id][2].ch0_y = row;
        check(tensix_pacr(s, pipe_id, row == face_rows - 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
              "PACR did not complete");
    }
}

static uint32_t encoded_one(uint32_t format) {
    return format == 0 ? 0x3f800000 : format == 1 ? 0x3c00 : 0x3f80;
}

static uint32_t encoded_three(uint32_t format) {
    return format == 0 ? 0x40400000 : format == 1 ? 0x4200 : 0x4040;
}

static void write_value(TensixState *s, uint32_t row, uint32_t col, bool dst32,
                        uint32_t format, uint32_t value) {
    if (dst32) {
        write_dst32b(s, row, col, dst_encode_fp32(value));
    } else {
        write_dst16b(s, row, col, format == 1 ? dst_encode_fp16(value) : dst_encode_bf16(value));
    }
}

static void run_case(uint32_t format, bool dst32, bool disable_flags, bool accumulate, uint32_t pattern) {
    TensixState *s = configure(format, dst32, disable_flags, accumulate);
    auto *sram = g_t_tiles[0].sram;
    std::memset(sram + output_addr - 16, 0xa5, 16 + 1024 + 16);
    uint32_t bytes = format == 0 ? 4 : 2;
    for (uint32_t i = 0; i < face_rows * ROW_SIZE; ++i) {
        if (bytes == 4) mem_wr<uint32_t>(sram + output_addr + bytes * i, 0x40000000);
        else mem_wr<uint16_t>(sram + output_addr + bytes * i, 0x4000);
    }
    for (uint32_t row = 0; row < face_rows; ++row) {
        for (uint32_t col = 0; col < ROW_SIZE; ++col) {
            // Pattern 0: actual zeros; 1: stale nonzero Dst invalidated by ZEROACC;
            // 2: alternating valid/nonvalid rows; 3: mixed positive, negative, zero datums.
            uint32_t value = pattern == 0 ? 0 : encoded_one(dst32 ? 0 : format);
            if (pattern == 3) {
                if (col % 3 == 1) value |= dst32 ? 0x80000000 : 0x8000;
                if (col % 3 == 2) value = 0;
            }
            write_value(s, row, col, dst32, format, value);
        }
        if (pattern == 1 || (pattern == 2 && row % 2 == 0)) {
            check(tensix_zeroacc(s, pipe_id, row, 0, 0, 0, 0), "ZEROACC did not complete");
        }
    }
    uint16_t dst_before[DST_ROWS][ROW_SIZE];
    bool validity_before[DST_ROWS];
    std::memcpy(dst_before, s->dst, sizeof(dst_before));
    std::memcpy(validity_before, s->dst_row_valid, sizeof(validity_before));
    pack_face(s);
    check(std::memcmp(dst_before, s->dst, sizeof(dst_before)) == 0, "PACR changed physical Dst data");
    check(std::memcmp(validity_before, s->dst_row_valid, sizeof(validity_before)) == 0,
          "output zero flag control changed Dst row validity");
    for (uint32_t row = 0; row < face_rows; ++row) {
        for (uint32_t col = 0; col < ROW_SIZE; ++col) {
            bool zero = pattern <= 1 || (pattern == 2 && row % 2 == 0) || (pattern == 3 && col % 3 == 2);
            bool negative = pattern == 3 && col % 3 == 1;
            uint32_t expected;
            if (accumulate) {
                expected = zero ? (format == 0 ? 0x40000000 : 0x4000) :
                           negative ? encoded_one(format) : encoded_three(format);
            } else {
                expected = zero ? 0 : encoded_one(format) | (negative ? (bytes == 4 ? 0x80000000 : 0x8000) : 0);
            }
            uint32_t addr = output_addr + bytes * (row * ROW_SIZE + col);
            uint32_t actual = bytes == 4 ? mem_rd<uint32_t>(sram + addr) : mem_rd<uint16_t>(sram + addr);
            if (actual != expected) {
                std::fprintf(stderr, "format=%u dst32=%u flag=%u acc=%u pattern=%u row=%u col=%u got=%08x expected=%08x\n",
                             format, dst32, disable_flags, accumulate, pattern, row, col, actual, expected);
                check(false, "packed numerical result differs");
            }
        }
    }
    for (uint32_t i = 0; i < 16; ++i) {
        check(sram[output_addr - 16 + i] == 0xa5, "PACR overwrote tile header");
        check(sram[output_addr + bytes * face_rows * ROW_SIZE + i] == 0xa5,
              "PACR overwrote the next face");
    }
}

static void run_transition() {
    TensixState *s = configure(5, true, false, false);
    // The baseline K-loop changes both controls after its initial overwrite.
    for (uint32_t row = 0; row < face_rows; ++row) {
        for (uint32_t col = 0; col < ROW_SIZE; ++col) {
            write_value(s, row, col, true, 5, 0x3f808000); // BF16 halfway: ties away -> 0x3f81.
        }
    }
    pack_face(s);
    for (uint32_t i = 0; i < face_rows * ROW_SIZE; ++i) {
        check(mem_rd<uint16_t>(g_t_tiles[0].sram + output_addr + 2 * i) == 0x3f81,
              "initial BF16 pack lost ties-away rounding");
    }
    s->config[0].THCON_SEC0_REG1_Disable_pack_zero_flags = 1;
    s->config[0].THCON_SEC0_REG1_Pack_L1_Acc = 1;
    check(tensix_zeroacc(s, pipe_id, 0, 0, 0, 0, 3), "ZEROACC all did not complete");
    pack_face(s);
    for (uint32_t i = 0; i < face_rows * ROW_SIZE; ++i) {
        check(mem_rd<uint16_t>(g_t_tiles[0].sram + output_addr + 2 * i) == 0x3f81,
              "zero chunk destroyed existing L1 accumulation");
    }
    s->config[0].THCON_SEC0_REG1_Disable_pack_zero_flags = 0;
    s->config[0].THCON_SEC0_REG1_Pack_L1_Acc = 0;
    pack_face(s);
    for (uint32_t i = 0; i < face_rows * ROW_SIZE; ++i) {
        check(mem_rd<uint16_t>(g_t_tiles[0].sram + output_addr + 2 * i) == 0,
              "return to overwrite retained stale L1 values");
    }
}

int main(int argc, char **argv) {
    if (argc == 2) {
        TensixState *s = configure(5, false, true, false);
        if (std::strcmp(argv[1], "compression") == 0) s->config[0].THCON_SEC0_REG1_Disable_zero_compress = 0;
        else if (std::strcmp(argv[1], "threshold") == 0) s->config[0].THCON_SEC0_REG1_Exp_threshold_en = 1;
        else if (std::strcmp(argv[1], "metadata") == 0) {
            tensix_setgpr(s, pipe_id, 0, 1, 8, 0); // SETDMAREG special: AllZeroFlags.
            check(false, "unsupported metadata read was accepted");
        } else check(false, "unknown negative case");
        pack_face(s);
        check(false, "unrelated unsupported feature was accepted");
    }
    for (uint32_t flag = 0; flag < 2; ++flag) {
        for (uint32_t acc = 0; acc < 2; ++acc) {
            for (uint32_t pattern = 0; pattern < 4; ++pattern) {
                run_case(5, false, flag, acc, pattern);
                run_case(1, false, flag, acc, pattern);
                run_case(5, true, flag, acc, pattern);
                run_case(0, true, flag, acc, pattern);
            }
        }
    }
    run_transition();
    std::puts("PASS: 64 full-face PACR cases, BF16 ties-away and K-loop transitions; Dst data/validity preserved");
}
