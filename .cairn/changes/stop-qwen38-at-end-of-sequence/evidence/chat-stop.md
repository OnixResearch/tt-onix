# Chat requests stop at end of sequence

Recorded 2026-09-25 on P150x2 with the default serving settings: fused GDN
decode step, grouped prefill, and traced prefill. Each question was sent as a
chat request with `max_tokens: 1024` and `temperature: 0`. The throwaway
harness is `.pi/tmp/gdn-step/service_chat.sh`.

- Before: package `/nix/store/pqasgm6hj8k32vhmss0xd0a1pm2gv34v-qwen38-0.1.0`.
- After: package `/nix/store/vd0k32r0z748rz0l1rx7isrbi2v32k1a-qwen38-0.1.0`.

| Question | Before: tokens, finish, wall | After: tokens, finish, wall |
| --- | --- | --- |
| What is 17 times 23? | 1024, length, 48.5 s | 60, stop, 3.6 s |
| Name three primary colors. | 1024, length, 48.3 s | 135, stop, 6.9 s |
| Say hello in French. | 1024, length, 48.1 s | 51, stop, 3.1 s |
| What is the capital of Japan? | 1024, length, 49.7 s | 20, stop, 1.7 s |
| All four concurrently (one batch) | 1024 each, 57.7 s | 60/135/51/20, stop, 7.8 s |

In all eight responses, the new text is a prefix of the old text. The old text
then continued past the end of turn with invented turns, such as
`"\n\nuser\n\nassistant\n<think>\nWe need answer user's simple math..."`.
