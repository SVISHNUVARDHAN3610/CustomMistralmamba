"""Model constants."""

# Bumped when memory-path NaN guards change. Logged by cloud train script so
# Kaggle runs can prove they imported this revision (not a stale notebook copy).
# -v4: assoc retrieval loss normalized/clamped (F.normalize + assoc_err_clip)
# and new assoc_state_norm_loss memory bound (lambda_assoc_norm); both bound
# the unbounded magnitudes that drove the step-2800/5400/10200/10600 blowups.
# -v3: explicit zeroing of all-masked SDPA query rows (SlidingWindowGQA);
# previously the GQA path relied on backend behavior for pad-only rows.
MEMORY_NAN_FIX_ID = (
    "all_masked_softmax+write_buffer_mask+masked_recon_gate+sdpa_all_masked"
    "+assoc_err_clip+assoc_state_norm-v4"
)
