"""Coverage for the regularization sample budget that bounds reg-image repeat balancing.

sd-scripts repeats the regularization subset until its sample count reaches the training sample
count. `regularization_to_training_sample_ratio` scales that target so regularization can be given
fewer optimizer steps than the training images without changing prior_loss_weight.
"""

from library.dreambooth_dataset import compute_regularization_sample_budget


def test_default_ratio_matches_the_training_sample_count():
    assert compute_regularization_sample_budget(84, 1.0) == 84
    assert compute_regularization_sample_budget(1, 1.0) == 1


def test_half_ratio_halves_the_budget():
    assert compute_regularization_sample_budget(84, 0.5) == 42


def test_fractional_budget_rounds_up():
    # 47 * 0.5 = 23.5 -> 24, so a ratio never silently drops below the requested share.
    assert compute_regularization_sample_budget(47, 0.5) == 24
    assert compute_regularization_sample_budget(10, 0.33) == 4


def test_ratio_above_one_increases_the_budget():
    assert compute_regularization_sample_budget(84, 2.0) == 168


def test_zero_ratio_disables_regularization_sampling():
    assert compute_regularization_sample_budget(84, 0.0) == 0


def test_negative_ratio_is_clamped_to_zero():
    assert compute_regularization_sample_budget(84, -1.0) == 0


def test_no_training_images_yields_no_budget():
    assert compute_regularization_sample_budget(0, 1.0) == 0
    assert compute_regularization_sample_budget(0, 0.5) == 0


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
    print("all regularization sample budget tests passed")
