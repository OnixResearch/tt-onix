## Why

The service generated exactly `max_tokens` tokens for every request and always
reported `finish_reason: "length"`. A chat answer ends with `<|im_end|>`. The
model then kept writing invented `user` and `assistant` turns until the bound.
Special tokens were stripped when decoding, so that text reached the client as
part of the answer. With `max_tokens: 1024`, a 20-token answer took about 48 s
of decode, and its text contained the invented turns.

## What Changes

- Read the end-of-sequence ids from the model's `generation_config.json`
  (`eos_token_id`, an id or a list) and from the tokenizer.
- The batch worker stops decoding once every user in the batch has produced a
  stop id. The serialized traced and eager loops stop at the first stop id.
- Each response keeps the tokens before the first stop id. It reports
  `finish_reason: "stop"`, or `"length"` when the bound was reached first.
  The MTP path truncates the same way.
- Batch-admission and trace-reuse admission measurements keep the fixed
  `max_tokens` length that their receipts assume.

## Impact

- **Files**: `pkgs/qwen36/service/qwen36_service/{core,batch_service,server}.py`
  and `pkgs/qwen36/check.py`.
- **Behavior**: responses end at the model's end of turn. Clients that relied on
  exactly `max_tokens` completion tokens now see fewer tokens and
  `finish_reason: "stop"`.
