# Negative fixture: repeated trace capture weakens the persistent recovery contract.
let profile = import "../mtp-rollback-profile.ncl" in
profile & { required_process_trace_capture_count = 2 }
