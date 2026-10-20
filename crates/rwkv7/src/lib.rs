//! RWKV-7 WKV7 recurrence: a pure host reference and a composed native gate.
//!
//! The equation is the token-at-a-time Goose recurrence (`wkv7_cuda.cu`):
//!
//! ```text
//! sa_i    = sum_j a[j] * S[i,j]
//! S'[i,j] = S[i,j] * w[j] + sa_i * b[j] + k[j] * v[i]
//! out_i   = sum_j S'[i,j] * r[j]
//! ```
//!
//! The reference is device-free. [`step_device`] expresses the same equation as
//! Burn tensor operations on `TtBackend`, exercising the matmul, broadcast,
//! elementwise and transfer paths. [`step_head_device_attached`] is the fused
//! route: one head through `burn_tt::wkv7_step` and the one-head Tensix kernel
//! (`tt_kernels::wkv7`), one device job.

pub mod layer;
pub mod tokenizer;

pub mod checkpoint;

use burn::tensor::{Tensor, TensorData, TensorPrimitive};
use burn_tt::{accurate_matmul, TtBackend, TtDevice, TtError};

pub const HEAD_COUNT: usize = 12;
pub const HEAD_SIZE: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct Inputs {
    /// `[head][D]`
    pub r: Vec<f32>,
    pub w: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    /// `[head][D][D]`
    pub state: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Outputs {
    /// `[head][D][D]`
    pub state: Vec<f32>,
    /// `[head][D]`
    pub out: Vec<f32>,
}

impl Inputs {
    pub fn validate(&self) -> Result<(), String> {
        let rows = HEAD_COUNT * HEAD_SIZE;
        for (name, values) in [
            ("r", &self.r),
            ("w", &self.w),
            ("k", &self.k),
            ("v", &self.v),
            ("a", &self.a),
            ("b", &self.b),
        ] {
            if values.len() != rows {
                return Err(format!(
                    "{name}: {} elements, expected {rows} (heads x head size)",
                    values.len()
                ));
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(format!("{name}: non-finite value"));
            }
        }
        if self.state.len() != HEAD_COUNT * HEAD_SIZE * HEAD_SIZE {
            return Err(format!(
                "state: {} elements, expected {} (heads x head size x head size)",
                self.state.len(),
                HEAD_COUNT * HEAD_SIZE * HEAD_SIZE
            ));
        }
        if self.state.iter().any(|value| !value.is_finite()) {
            return Err("state: non-finite value".into());
        }
        Ok(())
    }
}

/// One token of the WKV7 recurrence, in F32 host order.
pub fn step(inputs: &Inputs) -> Result<Outputs, String> {
    inputs.validate()?;
    let d = HEAD_SIZE;
    let rows = HEAD_COUNT * d;
    let mut next_state = vec![0.0_f32; inputs.state.len()];
    let mut out = vec![0.0_f32; rows];

    for head in 0..HEAD_COUNT {
        let head_base = head * d;
        let state_head = &inputs.state[head * d * d..(head + 1) * d * d];
        for i in 0..d {
            let mut sa = 0.0_f32;
            for j in 0..d {
                sa += inputs.a[head_base + j] * state_head[i * d + j];
            }
            for j in 0..d {
                next_state[head * d * d + i * d + j] = state_head[i * d + j]
                    * inputs.w[head_base + j]
                    + sa * inputs.b[head_base + j]
                    + inputs.k[head_base + j] * inputs.v[head_base + i];
            }
            let mut acc = 0.0_f32;
            for j in 0..d {
                acc += next_state[head * d * d + i * d + j] * inputs.r[head_base + j];
            }
            out[head_base + i] = acc;
        }
    }
    Ok(Outputs {
        state: next_state,
        out,
    })
}

fn vector2(values: &[f32], device: &TtDevice) -> Tensor<TtBackend, 2> {
    Tensor::from_data(
        TensorData::new(values.to_vec(), [HEAD_COUNT, HEAD_SIZE]),
        device,
    )
}

fn tensor3(values: &[f32], device: &TtDevice) -> Tensor<TtBackend, 3> {
    Tensor::from_data(
        TensorData::new(values.to_vec(), [HEAD_COUNT, HEAD_SIZE, HEAD_SIZE]),
        device,
    )
}

fn to_vec<const D: usize>(tensor: Tensor<TtBackend, D>) -> Result<Vec<f32>, String> {
    tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| error.to_string())
}

/// One token of the WKV7 recurrence on `device`, composed from Burn tensor
/// operations.
///
/// `accurate_matmul` keeps the matrix products near the exact F32 result; the
/// elementwise terms are native F32. The fused Tensix kernel remains future work.
pub fn step_device(device: TtDevice, inputs: &Inputs) -> Result<Outputs, String> {
    burn_tt::init(device, burn_tt::TtConfig::default())
        .map_err(|error| format!("attach card: {error}"))?;
    step_device_attached(device, inputs)
}

/// [`step_device`] for an already attached card.
pub fn step_device_attached(device: TtDevice, inputs: &Inputs) -> Result<Outputs, String> {
    inputs.validate()?;
    let state = tensor3(&inputs.state, &device);
    let a = vector2(&inputs.a, &device);
    let b = vector2(&inputs.b, &device);
    let w = vector2(&inputs.w, &device);
    let k = vector2(&inputs.k, &device);
    let v = vector2(&inputs.v, &device);
    let r = vector2(&inputs.r, &device);

    let (next_state, out) = accurate_matmul(|| {
        let sa = state.clone().matmul(a.unsqueeze_dim::<3>(2));
        let decay = state * w.unsqueeze_dim::<3>(1);
        let rank = sa * b.unsqueeze_dim::<3>(1);
        let outer = v.unsqueeze_dim::<3>(2) * k.unsqueeze_dim::<3>(1);
        let next = decay + rank + outer;
        let out = next
            .clone()
            .matmul(r.unsqueeze_dim::<3>(2))
            .squeeze_dim::<2>(2);
        (next, out)
    });

    Ok(Outputs {
        state: to_vec(next_state)?,
        out: to_vec(out)?,
    })
}

/// One head of the recurrence through the fused kernel
/// (`burn_tt::wkv7_step`): one device job instead of the composed path's
/// matmuls, broadcasts and adds.
///
/// The kernel returns one `[64, 96]` tensor -- the updated state in columns
/// 0..64 and the output replicated across 64..96 -- split by
/// [`tt_kernels::wkv7::split_step`]. `device` must already be attached.
pub struct HeadOutputs {
    /// `[64, 64]`, row-major.
    pub state: Vec<f32>,
    /// `[64]`.
    pub out: Vec<f32>,
}

/// [`HeadOutputs`] for `head` of `inputs`, on an attached `device`.
pub fn step_head_device_attached(
    device: TtDevice,
    inputs: &Inputs,
    head: usize,
) -> Result<HeadOutputs, String> {
    inputs.validate()?;
    if head >= HEAD_COUNT {
        return Err(format!("head {head} of {HEAD_COUNT}"));
    }
    let d = HEAD_SIZE;
    let base = head * d;
    let state_base = head * d * d;
    let float = |t: Tensor<TtBackend, 2>| match t.into_primitive() {
        TensorPrimitive::Float(t) => t,
        _ => unreachable!("an F32 tensor"),
    };
    let matrix = |values: Vec<f32>, rows: usize, cols: usize| {
        float(Tensor::from_data(
            TensorData::new(values, [rows, cols]),
            &device,
        ))
    };
    let state = matrix(inputs.state[state_base..state_base + d * d].to_vec(), d, d);
    let row = |v: &[f32]| matrix(v[base..base + d].to_vec(), 1, d);
    let combined = burn_tt::wkv7_step(
        state,
        row(&inputs.r),
        row(&inputs.w),
        row(&inputs.k),
        matrix(inputs.v[base..base + d].to_vec(), d, 1),
        row(&inputs.a),
        row(&inputs.b),
    );
    let values = Tensor::<TtBackend, 2>::from_primitive(TensorPrimitive::Float(combined))
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| error.to_string())?;
    let (state, out) = tt_kernels::wkv7::split_step(&values);
    Ok(HeadOutputs { state, out })
}

/// Attach a card once for a multi-step run. Returns the device on success.
pub fn attach(card: u16) -> Result<TtDevice, TtError> {
    let device = TtDevice::new(card);
    burn_tt::init(device, burn_tt::TtConfig::default())?;
    Ok(device)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = ((*seed >> 40) as f32) / ((1u32 << 24) as f32);
        unit * 2.0 - 1.0
    }

    fn inputs(seed: u64) -> Inputs {
        let mut state = seed;
        let mut values = |n: usize| (0..n).map(|_| lcg(&mut state) * 0.25).collect::<Vec<_>>();
        Inputs {
            r: values(HEAD_COUNT * HEAD_SIZE),
            w: values(HEAD_COUNT * HEAD_SIZE),
            k: values(HEAD_COUNT * HEAD_SIZE),
            v: values(HEAD_COUNT * HEAD_SIZE),
            a: values(HEAD_COUNT * HEAD_SIZE),
            b: values(HEAD_COUNT * HEAD_SIZE),
            state: values(HEAD_COUNT * HEAD_SIZE * HEAD_SIZE),
        }
    }

    #[test]
    fn recurrence_terms_are_all_present() {
        let input = inputs(7);
        let first = step(&input).unwrap();
        let mut zero_decay = input.clone();
        zero_decay.w = vec![0.0; zero_decay.w.len()];
        let second = step(&zero_decay).unwrap();
        assert_ne!(first.state, second.state);
        assert_ne!(first.out, second.out);

        let mut zero_rank = input.clone();
        zero_rank.b = vec![0.0; zero_rank.b.len()];
        assert_ne!(step(&zero_rank).unwrap().state, first.state);

        let mut zero_outer = input.clone();
        zero_outer.v = vec![0.0; zero_outer.v.len()];
        assert_ne!(step(&zero_outer).unwrap().state, first.state);
    }

    #[test]
    fn state_carries_into_the_next_token() {
        let first_input = inputs(11);
        let first = step(&first_input).unwrap();
        let mut second_input = inputs(13);
        second_input.state = first.state.clone();
        let second = step(&second_input).unwrap();
        assert_ne!(first.state, second.state);
        assert_ne!(first.out, second.out);
    }
}
