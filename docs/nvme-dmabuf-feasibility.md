# NVMe-to-Blackhole DMA-BUF feasibility

## Result

**Blocked at the kernel UAPI boundary on this host.** No device, NVMe
namespace, DMA-BUF, io_uring, reset, or transfer was performed.

The packaged tt-kmd UAPI exposes `TENSTORRENT_IOCTL_EXPORT_TLB_DMABUF`, but the
running Linux 6.18.50 UAPI has no io_uring DMA-BUF registration symbol. The
user-space path needed to import a page-less exported TLB window cannot be
built against this kernel as shipped. There is also no caller-provisioned
disposable NVMe namespace.

This is the "reproducible blocked/unsupported result" branch of the issue, not
a hardware failure or a throughput result.

## Source-bound capability matrix

| Stage | Observation | Status |
| --- | --- | --- |
| Kernel config | `CONFIG_IO_URING=y`, `CONFIG_DMABUF_MOVE_NOTIFY=y` | available |
| Kernel io_uring DMA-BUF import UAPI | no `IORING_REGISTER_DMABUF`/registration symbol in the 6.18.50 UAPI | **absent** |
| tt-kmd TLB export UAPI | `TENSTORRENT_IOCTL_EXPORT_TLB_DMABUF` present | available |
| TLB export runtime | not exercised | blocked |
| NVMe import path | not exercised | blocked |
| Disposable NVMe namespace | not supplied | blocked |
| Transfer / byte verification | not attempted | blocked |

## Reproduction

```sh
python3 scripts/nvme-dmabuf-capability.py \
  --kernel-uapi /nix/store/2cvzgrdv723ymf99vv62z86f84iqxam9-linux-6.18.50-dev/lib/modules/6.18.50/source/include/uapi/linux/io_uring.h \
  --kmd-uapi .pi/references/tenstorrent/tt-kmd/ioctl.h --json
```

Observed result: `status: blocked`; reasons are the missing io_uring DMA-BUF
registration and the absent disposable namespace. The script reads only
configuration and headers; it never opens a device or namespace.

If the matched kernel source lives elsewhere, pass its
`include/uapi/linux/io_uring.h`. The KMD path is the package-matched
`tt-kmd` UAPI, not a version string.

## Ownership/lifecycle boundary

If a future kernel carries the import UAPI, the permitted sequence is:

1. reserve destination TT scratch and keep it owned for the whole operation;
2. allocate/configure the TLB window, export it, and keep the export alive;
3. register/import it with io_uring and submit only reads;
4. verify every byte and untouched guard region through an independent
   readback;
5. release the io_uring registration only after completion, then the export,
   then the TLB window, then the destination.

No visible `/dev/tenstorrent` fd is proof of quiescence: KMD retains exported
TLB windows while an importer holds them and refuses reset with `-EBUSY`. A
timeout or cancellation is not proof that DMA stopped. Automatic reset,
destructive fault injection, and host-disk discovery are out of scope.

## Decision

Stop this feasibility path on the current kernel. Revisit only when:

- the running kernel and UAPI carry the exact io_uring DMA-BUF registration
  used by the probe, and the kernel source/config is pinned with it;
- an isolated disposable NVMe namespace is provided;
- the device owner authorizes a bounded read-only probe.

No production loading, model-server, checkpoint-streaming, or recurrent-state
change follows from this report.
