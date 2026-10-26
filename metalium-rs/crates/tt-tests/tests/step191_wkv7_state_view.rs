//! F32 WKV state consumption from a combined-output parent, without a state copy.
use std::time::Instant;

use tt_kernels::session::TileChoice;
use tt_kernels::tensor::{BlockMove, DramTensor};
use tt_kernels::wkv7::{HEADS, HEAD_SIZE, STEP_COLS};
use tt_tests::backend::{with_session, GATE_TILE, ON_SILICON};

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn combined_parent_state_is_consumed_without_materialization() {
    with_session(TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1), |s| {
        let rows = HEADS * HEAD_SIZE;
        // Adjacent output columns are deliberately poisonous: a two-column tile
        // pitch on this three-column parent contaminates the recurrence.
        let data: Vec<f32> = (0..rows * STEP_COLS)
            .map(|i| {
                if i % STEP_COLS >= HEAD_SIZE {
                    f32::from_bits(0x7fc1_2345)
                } else {
                    ((i * 17 % 251) as f32 - 125.0) / 1024.0
                }
            })
            .collect();
        let parent = s.upload(&data, rows, STEP_COLS).unwrap();
        let row = |salt: usize| -> Vec<f32> {
            (0..rows)
                .map(|i| ((i * salt % 97) as f32 - 48.0) / 256.0)
                .collect()
        };
        let a = s.upload(&row(3), 1, rows).unwrap();
        let b = s.upload(&row(5), 1, rows).unwrap();
        let w = s.upload(&row(7), 1, rows).unwrap();
        let k = s.upload(&row(11), 1, rows).unwrap();
        let r = s.upload(&row(13), 1, rows).unwrap();
        let columns: Vec<DramTensor> = (0..HEADS)
            .map(|h| s.upload(&row(19)[h * 64..h * 64 + 64], 64, 1).unwrap())
            .collect();
        let v: Vec<_> = columns.iter().collect();
        let moves = [BlockMove {
            from: [0, 0],
            to: [0, 0],
            extent: [rows, HEAD_SIZE],
            transposed: false,
        }];
        let compact = s.copy_blocks(&parent, &moves, [rows, HEAD_SIZE]).unwrap();
        let expected = s
            .wkv7_step_batched(&compact, &a, &b, &w, &k, &r, &v)
            .unwrap();
        let want = s.download(&expected).unwrap();
        s.free(expected).unwrap();
        s.free(compact).unwrap();
        let before = s.dataflow_stats().clone();
        let direct = s
            .wkv7_step_batched(&parent, &a, &b, &w, &k, &r, &v)
            .unwrap();
        let after = s.dataflow_stats().clone();
        let got = s.download(&direct).unwrap();
        assert_eq!(
            bits(&got),
            bits(&want),
            "wide-parent and compact consumers differ"
        );
        assert!(
            got.iter().all(|x| x.is_finite()),
            "excluded NaNs entered the state"
        );
        assert_eq!(
            after.transfer_packets, before.transfer_packets,
            "direct consumer copied state"
        );
        assert_eq!(
            bits(&s.download(&parent).unwrap()),
            bits(&data),
            "consumer mutated parent"
        );
        s.free(direct).unwrap();

        let mut special = data.clone();
        for (i, word) in [0x8000_0000, 1, 0x7fc1_2345, 0x7f80_0000, 0xff80_0000]
            .into_iter()
            .enumerate()
        {
            special[i] = f32::from_bits(word);
        }
        let special_parent = s.upload(&special, rows, STEP_COLS).unwrap();
        let compact = s
            .copy_blocks(&special_parent, &moves, [rows, HEAD_SIZE])
            .unwrap();
        let baseline = s
            .wkv7_step_batched(&compact, &a, &b, &w, &k, &r, &v)
            .unwrap();
        let direct = s
            .wkv7_step_batched(&special_parent, &a, &b, &w, &k, &r, &v)
            .unwrap();
        let baseline_values = s.download(&baseline).unwrap();
        let direct_values = s.download(&direct).unwrap();
        assert_eq!(
            bits(&direct_values),
            bits(&baseline_values),
            "live special values changed"
        );
        assert!(
            direct_values.iter().any(|x| x.is_nan()),
            "live NaN was lost"
        );
        assert_eq!(bits(&s.download(&special_parent).unwrap()), bits(&special));
        for t in [direct, baseline, compact, special_parent] {
            s.free(t).unwrap();
        }

        let reps = if ON_SILICON { 7 } else { 1 };
        let mut medians = Vec::new();
        for materialize in [true, false] {
            let mut times = Vec::new();
            let before = s.dataflow_stats().clone();
            for iteration in 0..reps + 2 {
                let start = Instant::now();
                let compact =
                    materialize.then(|| s.copy_blocks(&parent, &moves, [rows, HEAD_SIZE]).unwrap());
                let out = s
                    .wkv7_step_batched(compact.as_ref().unwrap_or(&parent), &a, &b, &w, &k, &r, &v)
                    .unwrap();
                s.sync().unwrap();
                let elapsed = start.elapsed();
                // Validate every timed result, outside the measured region.
                assert_eq!(bits(&s.download(&out).unwrap()), bits(&want));
                s.free(out).unwrap();
                if let Some(compact) = compact {
                    s.free(compact).unwrap();
                }
                if iteration >= 2 {
                    times.push(elapsed);
                }
            }
            times.sort_unstable();
            let median = times[times.len() / 2];
            medians.push(median);
            println!("WKV_STATE_VIEW simulated={} materialize={materialize} reps={reps} median_us={} transfer_packets={} batches={}",
                !ON_SILICON, median.as_micros(),
                s.dataflow_stats().transfer_packets - before.transfer_packets,
                s.dataflow_stats().batches - before.batches);
        }
        println!(
            "WKV_STATE_VIEW ratio={:.6}",
            medians[1].as_secs_f64() / medians[0].as_secs_f64()
        );
    });
}

#[test]
fn burn_state_slice_carries_two_tokens_without_a_copy_when_opted_in() {
    use burn::tensor::{backend::Backend, Tensor, TensorData, TensorPrimitive};
    use burn_tt::{TtBackend, TtTensor};
    use tt_tests::burn_device::{with_device, Config};

    fn primitive<const D: usize>(t: Tensor<TtBackend, D>) -> TtTensor {
        let TensorPrimitive::Float(p) = t.into_primitive() else {
            unreachable!()
        };
        p
    }
    fn state(parent: &Tensor<TtBackend, 2>, transpose: bool) -> TtTensor {
        let state = parent.clone().slice([0..768, 0..64]).reshape([12, 64, 64]);
        primitive(if transpose {
            state.swap_dims(1, 2)
        } else {
            state
        })
    }
    fn copies() -> u64 {
        burn_tt::device_time()
            .iter()
            .filter(|(op, _, _)| matches!(*op, "copy_blocks" | "repack" | "repack_many"))
            .map(|(_, jobs, _)| *jobs)
            .sum()
    }
    with_device(Config::default(), |device| {
        let make = |data: Vec<f32>, dims: [usize; 2]| {
            let p = primitive(Tensor::<TtBackend, 2>::from_data(
                TensorData::new(data, dims),
                &device,
            ));
            p.ensure_resident();
            p
        };
        let data: Vec<_> = (0..768 * 96)
            .map(|i| {
                if i % 96 >= 64 {
                    f32::from_bits(0x7fc1_2345)
                } else {
                    ((i * 17 % 127) as f32 - 63.0) / 256.0
                }
            })
            .collect();
        let mut parent =
            Tensor::<TtBackend, 2>::from_primitive(TensorPrimitive::Float(make(data, [768, 96])));
        let row = |salt| {
            (0..768)
                .map(|i| ((i * salt % 97) as f32 - 48.0) / 256.0)
                .collect()
        };
        let a = make(row(3), [1, 768]);
        let b = make(row(5), [1, 768]);
        let w = make(row(7), [1, 768]);
        let k = make(row(11), [1, 768]);
        let r = make(row(13), [1, 768]);
        let v: Vec<_> = (0..12)
            .map(|h| make(vec![(h as f32 + 1.0) / 128.0; 64], [64, 1]))
            .collect();
        // The third step is deliberately outside the supported layout. It must
        // still materialize, not silently treat a transposed state as an alias.
        for token in 0..3 {
            let compact = state(&parent, token == 2);
            compact.ensure_resident();
            let expected = burn_tt::wkv7_step_batched(
                compact,
                a.clone(),
                b.clone(),
                w.clone(),
                k.clone(),
                r.clone(),
                v.clone(),
            );
            let expected = Tensor::<TtBackend, 2>::from_primitive(TensorPrimitive::Float(expected));
            let want: Vec<f32> = expected.into_data().to_vec().unwrap();
            let before = copies();
            let traffic = burn_tt::tensor_traffic();
            let result = burn_tt::wkv7_step_batched(
                state(&parent, token == 2),
                a.clone(),
                b.clone(),
                w.clone(),
                k.clone(),
                r.clone(),
                v.clone(),
            );
            assert!(result.computed_on_device());
            TtBackend::sync(&device).unwrap();
            assert_eq!(burn_tt::tensor_traffic().downloads, traffic.downloads);
            let copied = copies() - before;
            let enabled = std::env::var("TT_WKV7_STATE_VIEW").as_deref() == Ok("1");
            assert_eq!(
                copied,
                u64::from(!enabled || token == 2),
                "token {token}: incorrect materialization count"
            );
            parent = Tensor::<TtBackend, 2>::from_primitive(TensorPrimitive::Float(result));
            let got: Vec<f32> = parent.clone().into_data().to_vec().unwrap();
            assert_eq!(
                bits(&got),
                bits(&want),
                "token {token}: routed state consumer differs"
            );
            assert!(got.iter().all(|x| x.is_finite()));
            println!("WKV_STATE_VIEW_BURN token={token} enabled={enabled} state_materializations={copied} downloads=0");
        }
    });
}
