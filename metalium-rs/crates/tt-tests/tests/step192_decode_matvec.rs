//! M=1 live-row kernels preserve the original Matrix Unit accumulation.
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_tests::harness::BUDGET;

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    tt_ttsim::fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(2),
            |_, _| Ok(None),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    })
    .unwrap();
}

#[cfg(feature = "silicon")]
fn with_session(f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    tt_ttsim::fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(12),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    })
    .unwrap();
}

fn same(a: &[f32], b: &[f32]) {
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "element {i}: {a} vs {b}");
    }
    assert_eq!(a.len(), b.len());
}

#[test]
fn decode_matches_independent_integer_oracle_and_fractional_baseline() {
    with_session(|s| {
        for fidelity in [
            Fidelity::Lo,
            Fidelity::HiFi2,
            Fidelity::HiFi3,
            Fidelity::HiFi4,
        ] {
            for transposed in [false, true] {
                let (k, n) = (65, 35);
                let av: Vec<_> = (0..k).map(|i| (i % 7) as f32 - 3.).collect();
                let bv: Vec<_> = (0..k * n)
                    .map(|i| {
                        let (r, c) = if transposed {
                            (i % k, i / k)
                        } else {
                            (i / n, i % n)
                        };
                        if r == c {
                            1.
                        } else {
                            0.
                        }
                    })
                    .collect();
                let a = s.upload(&av, 1, k).unwrap();
                let b = s
                    .upload(
                        &bv,
                        if transposed { n } else { k },
                        if transposed { k } else { n },
                    )
                    .unwrap();
                s.set_decode_matvec(true);
                for limit in [None, std::num::NonZeroUsize::new(1)] {
                    s.set_matmul_k_block_limit(limit);
                    let out = s
                        .matmul_dram(
                            &a,
                            false,
                            &b,
                            transposed,
                            SrcRoute::Tf32FromFp32,
                            fidelity,
                            BUDGET * 40,
                        )
                        .unwrap();
                    same(&s.download(&out).unwrap(), &av[..n]);
                    s.free(out).unwrap();
                }
                s.set_matmul_k_block_limit(None);
                s.free(a).unwrap();
                s.free(b).unwrap();
                let av: Vec<_> = (0..k).map(|i| (i % 11) as f32 / 13. - 0.4).collect();
                let bv: Vec<_> = (0..k * n).map(|i| (i % 17) as f32 / 19. - 0.4).collect();
                let a = s.upload(&av, 1, k).unwrap();
                let b = s
                    .upload(
                        &bv,
                        if transposed { n } else { k },
                        if transposed { k } else { n },
                    )
                    .unwrap();
                for route in [SrcRoute::Tf32FromFp32, SrcRoute::Bf16FromFp32] {
                    s.set_decode_matvec(false);
                    let plain = s
                        .matmul_dram(&a, false, &b, transposed, route, fidelity, BUDGET * 40)
                        .unwrap();
                    let expected = s.download(&plain).unwrap();
                    s.set_decode_matvec(true);
                    let out = s
                        .matmul_dram(&a, false, &b, transposed, route, fidelity, BUDGET * 40)
                        .unwrap();
                    same(&s.download(&out).unwrap(), &expected);
                    s.free(plain).unwrap();
                    s.free(out).unwrap();
                }
                s.free(a).unwrap();
                s.free(b).unwrap();
            }
        }
    });
}

#[cfg(feature = "silicon")]
#[test]
fn rwkv_projection_and_lm_head_warmed_medians() {
    with_session(|s| {
        for (k, n) in [(768, 768), (768, 3072), (3072, 768), (768, 65536)] {
            let av: Vec<_> = (0..k).map(|i| (i % 7) as f32 - 3.).collect();
            // Real checkpoint orientation [N,K], no pretranspose/storage change.
            let bv: Vec<_> = (0..k * n)
                .map(|i| if i % k == (i / k) % k { 1. } else { 0. })
                .collect();
            let want: Vec<_> = (0..n).map(|i| av[i % k]).collect();
            let a = s.upload(&av, 1, k).unwrap();
            let b = s.upload(&bv, n, k).unwrap();
            for candidate in [false, true] {
                s.set_decode_matvec(candidate);
                let mut samples = Vec::new();
                for iteration in 0..9 {
                    let start = std::time::Instant::now();
                    let out = s
                        .matmul_dram(
                            &a,
                            false,
                            &b,
                            true,
                            SrcRoute::Tf32FromFp32,
                            Fidelity::Lo,
                            BUDGET * 100,
                        )
                        .unwrap();
                    s.sync().unwrap();
                    let elapsed = start.elapsed();
                    same(&s.download(&out).unwrap(), &want);
                    s.free(out).unwrap();
                    if iteration >= 2 {
                        samples.push(elapsed);
                    }
                }
                samples.sort();
                println!(
                    "decode_matvec candidate={candidate} shape=1x{k}x{n} median_us={:.3}",
                    samples[samples.len() / 2].as_secs_f64() * 1e6
                );
            }
            s.free(a).unwrap();
            s.free(b).unwrap();
        }
    });
}
