# face-shape

Estimates your face shape and pupillary distance from your webcam or a photo, to help pick glasses. Everything runs on your machine. No image leaves it.

The only network call is a one-time download of Google's MediaPipe face landmark model. After that it works offline.

## Run

Needs [uv](https://docs.astral.sh/uv/).

```sh
uv run face_shape.py              # live webcam, press q to finish
uv run face_shape.py photo.jpg    # a photo instead
uv run face_shape.py --men        # or --women, for one set of frame suggestions
uv run face_shape.py --check      # self-check
```

On macOS, allow camera access for your terminal when asked.

## How accurate is it?

Rough. The face shape cut-offs are guesses (tweak `LONG`, `WIDE` and `NARROW` at the top of the script), so faces near a boundary flip between two shapes.

A camera can't measure size on its own, so millimetres come from assuming your iris is 11.7 mm across. Real irises vary by about 5%, so PD is only good to a few millimetres. An optician will measure it properly.

It doesn't estimate face width: the widest points of your face sit a few cm behind your eyes, so a single camera reads them too small. For frame size, use the numbers printed on the arm of glasses that already fit you, e.g. `52□18 140`.
