"""Weights & Biases EventWriter for detectron2.

detectron2 ships exactly three writers in detectron2/utils/events.py --
JSONWriter, TensorboardXWriter, CommonMetricPrinter -- and nothing for W&B.
This is the missing fourth.

It mirrors TensorboardXWriter (events.py:141) deliberately: same smoothing
window, same `_last_write` guard so an iteration is never logged twice, same
responsibility for draining storage's accumulated images. Anyone who can read
detectron2's own writer can read this one.
"""

from detectron2.utils.events import EventWriter, get_event_storage


class WandbWriter(EventWriter):
    """Write detectron2's scalar metrics to the active W&B run.

    Args:
        window_size: median-smoothing window, matching the other writers.
        log_images: forward storage's visualization images to W&B. Off by
            default -- these are large and detectron2 only populates them if
            something explicitly calls storage.put_image().
    """

    def __init__(self, window_size: int = 20, log_images: bool = False):
        self._window_size = window_size
        self._log_images = log_images
        self._last_write = -1

    @property
    def _run(self):
        import wandb

        if wandb.run is None:
            raise RuntimeError(
                "wandb.init() must be called before training starts; "
                "WandbWriter attaches to the active run."
            )
        return wandb.run

    def write(self):
        storage = get_event_storage()

        metrics = {}
        new_last_write = self._last_write
        for k, (v, iteration) in storage.latest_with_smoothing_hint(
            self._window_size
        ).items():
            if iteration > self._last_write:
                metrics[k] = v
                new_last_write = max(new_last_write, iteration)

        if metrics:
            self._run.log(metrics, step=new_last_write)
        self._last_write = new_last_write

        # storage accumulates images until a writer drains them. If we do not
        # clear, they grow without bound. Same contract TensorboardXWriter has.
        if len(storage._vis_data) >= 1:
            if self._log_images:
                import wandb

                images = {}
                for name, img, _ in storage._vis_data:
                    # detectron2 stores CHW; wandb.Image wants HWC.
                    if hasattr(img, "shape") and len(img.shape) == 3 and img.shape[0] in (1, 3):
                        img = img.transpose(1, 2, 0)
                    images[name] = wandb.Image(img)
                self._run.log(images, step=self._last_write)
            storage.clear_images()

    def close(self):
        # The run's lifecycle belongs to the caller (wandb.init/finish),
        # not to the writer. Closing it here would break multi-stage scripts.
        pass
