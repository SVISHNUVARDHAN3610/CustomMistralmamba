"""Model constants."""

# Bumped when memory-path NaN guards change. Logged by cloud train script so
# Kaggle runs can prove they imported this revision (not a stale notebook copy).
# -v3: explicit zeroing of all-masked SDPA query rows (SlidingWindowGQA);
# previously the GQA path relied on backend behavior for pad-only rows.
MEMORY_NAN_FIX_ID = (
    "all_masked_softmax+write_buffer_mask+masked_recon_gate+sdpa_all_masked-v3"
)
