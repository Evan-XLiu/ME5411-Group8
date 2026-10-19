"""Exercise both architectures through training and checkpoint persistence."""
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
from PIL import Image
from task_05.cnn import build_model, train
from common.datasets import stratified_split


def test_course_split_is_75_25():
    labels = np.repeat([0, 1], 8)
    training, testing = stratified_split(labels)
    assert len(training) == 12
    assert len(testing) == 4
    assert set(training).isdisjoint(testing)
    assert np.bincount(labels[testing]).tolist() == [2, 2]


@pytest.mark.parametrize("architecture", ["baseline", "yjn"])
def test_train_and_reload_both_variants(tmp_path, architecture):
    torch.set_num_threads(1)
    for label in range(2):
        directory = tmp_path / "data" / str(label)
        directory.mkdir(parents=True)
        for index in range(4):
            Image.fromarray(np.full((32, 32), label * 180 + index, dtype=np.uint8)).save(directory / f"{index}.png")
    checkpoint = tmp_path / "model.pt"
    metrics = train(tmp_path / "data", checkpoint, epochs=1, batch_size=2,
                    architecture=architecture, device="cpu")
    saved = torch.load(checkpoint, weights_only=True)
    restored = build_model(2, architecture=saved["architecture"])
    restored.load_state_dict(saved["state_dict"])
    restored.eval()
    assert restored(torch.zeros(2, 1, 32, 32)).shape == (2, 2)
    assert saved["test_fraction"] == 0.25
    assert metrics["train_samples"] == 6
    assert metrics["test_samples"] == 2
    assert 0 <= metrics["history"][0]["test_accuracy"] <= 1
