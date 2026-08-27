"""Header ablation: same drive, header on vs off."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reveal.header import (
    HEADER_CSV_FIELDS,
    HeaderIdentityError,
    HeaderRun,
    assert_header_changed,
    run_ablation,
    run_header,
    write_header_csv,
)


def test_headed_and_unheaded_differ():
    headed, unheaded = run_ablation(steps=4, seed=0, n_sites=4)
    assert headed.headed is True
    assert unheaded.headed is False
    assert headed.seed == unheaded.seed
    assert headed.steps == unheaded.steps
    assert headed.kappa == unheaded.kappa
    assert not np.allclose(headed.final_q, unheaded.final_q, rtol=0.0, atol=1e-15)


def test_header_metrics_present():
    headed, unheaded = run_ablation(steps=4, seed=1, n_sites=4)
    for row in (headed, unheaded):
        assert row.burst_count >= 0
        assert np.isfinite(row.mean_Theta)
        assert np.isfinite(row.mean_Theta_late)
        assert np.isfinite(row.rms_dTheta)
        assert np.isfinite(row.rms_dTheta_late)
        assert np.isfinite(row.residual_R)
        assert row.theta_crit > 0.0
        assert row.burn_in_steps == 0  # tiny N keeps the whole window
        assert row.wg_lock is False
        assert row.phi_b_lock is False
        assert row.wg_target == pytest.approx(111.408)
        assert row.phi_b_target == pytest.approx(0.8145)


def test_identity_fail_when_q_match():
    headed, unheaded = run_ablation(steps=4, seed=0, n_sites=4)
    fake_unheaded = HeaderRun(
        **{**unheaded.__dict__, "final_q": np.array(headed.final_q, copy=True)}
    )
    with pytest.raises(HeaderIdentityError):
        assert_header_changed(headed, fake_unheaded)
    assert_header_changed(headed, unheaded)


def test_write_header_csv_two_rows(tmp_path: Path):
    headed, unheaded = run_ablation(steps=4, seed=0, n_sites=4)
    path = write_header_csv([headed, unheaded], tmp_path / "header.csv")
    text = path.read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines[0].split(",") == HEADER_CSV_FIELDS
    assert len(lines) == 3
    assert "True" in lines[1] or "true" in lines[1].lower()
    assert "False" in lines[2] or "false" in lines[2].lower()


def test_same_seed_same_unheaded():
    a = run_header(headed=False, steps=4, seed=7, n_sites=4)
    b = run_header(headed=False, steps=4, seed=7, n_sites=4)
    np.testing.assert_allclose(a.final_q, b.final_q)
